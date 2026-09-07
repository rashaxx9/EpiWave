from __future__ import annotations

import gc
from collections import deque

from agents.respond import stream_response
from agents.retrieve import retrieve_chunks
from agents.verify import verify_chunks
from config import EEG_SCANS_PATH, MAX_HISTORY_TURNS
from eeg.analyzer import analyze_eeg
from eeg.parser import list_edf_files, parse_edf
from eeg.visualizer import generate_chart_data
from rag.ingest import run_ingestion_from_analysis
from rag.vectorize import clear_vector_store


def _run_analyze_pipeline() -> None:
    """Run the full EEG pipeline: parse → analyze → ingest into vector store."""
    edf_files = list_edf_files(EEG_SCANS_PATH)
    if not edf_files:
        print(f"No .edf files found in {EEG_SCANS_PATH}/")
        print("Place your EEG scan files (.edf) in that directory and try again.")
        return

    # Show available files with numbering so the user can pick
    print(f"\nFound {len(edf_files)} EEG scan(s):\n")
    for idx, edf_path in enumerate(edf_files, start=1):
        # Show parent folder for clarity (e.g. "chb01/chb01_03.edf")
        relative = edf_path.relative_to(EEG_SCANS_PATH)
        size_mb = edf_path.stat().st_size / (1024 * 1024)
        print(f"  [{idx:>3}] {relative}  ({size_mb:.1f} MB)")

    print(f"\nEnter file numbers to analyze (e.g. '1', '1-3', '1,4,7', or 'all'):")
    try:
        selection = input("Selection> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAnalysis cancelled.")
        return

    # Parse user selection into a list of indices
    selected_indices: list[int] = []
    if selection == "all":
        selected_indices = list(range(len(edf_files)))
    else:
        for part in selection.replace(" ", "").split(","):
            if "-" in part:
                bounds = part.split("-", 1)
                try:
                    start, end = int(bounds[0]), int(bounds[1])
                    selected_indices.extend(range(start - 1, end))
                except ValueError:
                    print(f"  Invalid range: '{part}'. Skipping.")
            else:
                try:
                    selected_indices.append(int(part) - 1)
                except ValueError:
                    print(f"  Invalid number: '{part}'. Skipping.")

    # Validate indices
    selected_files = []
    for idx in selected_indices:
        if 0 <= idx < len(edf_files):
            selected_files.append(edf_files[idx])
        else:
            print(f"  Index {idx + 1} out of range. Skipping.")

    if not selected_files:
        print("No valid files selected. Returning to chat.")
        return

    print("\nClearing previous scans from vector store...")
    clear_vector_store()

    print(f"\nProcessing {len(selected_files)} file(s)...\n")

    for edf_path in selected_files:
        relative = edf_path.relative_to(EEG_SCANS_PATH)
        print(f"--- Processing: {relative} ---")

        # Step 1: Parse the raw .edf file
        try:
            parsed = parse_edf(edf_path)
            print(f"  Parsed: {parsed['num_channels']} channels, "
                  f"{parsed['duration_seconds']:.1f}s, "
                  f"{parsed['sampling_frequency']}Hz")
        except Exception as exc:
            print(f"  Parse error: {exc}")
            continue

        # Step 2: Analyze for seizure patterns
        try:
            analysis = analyze_eeg(parsed)
            print(f"  Analysis: {analysis['spike_count']} spikes, "
                  f"{len(analysis['seizure_events'])} candidate seizure events")
        except Exception as exc:
            print(f"  Analysis error: {exc}")
            continue

        # Step 3: Generate chart data (stored for future web UI use)
        try:
            chart_data = generate_chart_data(parsed, analysis)
            print(f"  Chart data ready: {chart_data['metadata']['seizure_event_count']} "
                  f"seizure annotations, {len(chart_data['spike_markers'])} spike channels")
        except Exception as exc:
            print(f"  Visualizer warning: {exc}")
            # Non-fatal — we can still ingest the analysis without chart data

        # Step 4: Ingest analysis results into the vector store
        try:
            doc_count, chunk_count = run_ingestion_from_analysis(analysis)
            print(f"  Ingested: {doc_count} document(s), {chunk_count} chunks into vector store")
        except Exception as exc:
            print(f"  Ingestion error: {exc}")
            continue

        del parsed
        del analysis
        if 'chart_data' in locals():
            del chart_data
        gc.collect()

        print()

    print("EEG analysis pipeline complete. You can now ask questions about the scan(s).")


def main() -> None:
    history: deque[tuple[str, str]] = deque(maxlen=MAX_HISTORY_TURNS)

    print("=" * 60)
    print("  EpiWave — AI-Powered EEG Analysis Assistant")
    print("  For educational and informational purposes only.")
    print("=" * 60)
    print("\nCommands:")
    print("  'analyze'  — process EEG scans from data/eeg_scans/")
    print("  'quit'     — exit the application")
    print()

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        command = user_input.lower()
        if command == "quit":
            print("Goodbye.")
            break
        if command in ("analyze", "ingest"):
            _run_analyze_pipeline()
            continue

        try:
            retrieved_chunks = retrieve_chunks(user_input)
            verified_chunks = verify_chunks(retrieved_chunks)

            # Check if the vector store has ANY data so the LLM knows whether
            # scans have been analyzed (even if the current query didn't match).
            from rag.vectorize import _get_chroma_collection
            has_scans = _get_chroma_collection().count() > 0

            print("EpiWave> ", end="", flush=True)
            response_tokens = []
            for token in stream_response(verified_chunks, list(history), user_input, has_scans=has_scans):
                print(token, end="", flush=True)
                response_tokens.append(token)
            print()

            assistant_response = "".join(response_tokens).strip()
            history.append((user_input, assistant_response))
        except Exception as exc:
            print(f"Request failed: {exc}")


if __name__ == "__main__":
    main()
