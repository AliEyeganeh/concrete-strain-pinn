import shutil
from pathlib import Path

def ensure_copy(run: Path, src_name: str, dst_name: str):
    src = run / src_name
    dst = run / dst_name
    if dst.exists():
        return
    if src.exists():
        shutil.copyfile(src, dst)
        print(f"[+] {dst_name}  (copied from {src_name})")

def main():
    run = Path("out_ecc_scc_mcdo")  # change if your RUN is different

    # We already built these (from your last step):
    #   coverage_curve_temp_hist_test_raw.csv
    #   coverage_curve_iso_hist_test_raw.csv
    # Now create all common aliases your plotter might request.
    pairs = [
        # test set
        ("coverage_curve_temp_hist_test_raw.csv", "coverage_curve_temp_hist_test_temp.csv"),
        ("coverage_curve_iso_hist_test_raw.csv",  "coverage_curve_iso_hist_test_temp.csv"),
        # val set (some scripts ask for val even if you only have test; alias to test)
        ("coverage_curve_temp_hist_test_raw.csv", "coverage_curve_temp_hist_val_raw.csv"),
        ("coverage_curve_temp_hist_test_raw.csv", "coverage_curve_temp_hist_val_temp.csv"),
        ("coverage_curve_iso_hist_test_raw.csv",  "coverage_curve_iso_hist_val_raw.csv"),
        ("coverage_curve_iso_hist_test_raw.csv",  "coverage_curve_iso_hist_val_temp.csv"),
    ]

    made_any = False
    for src, dst in pairs:
        before = (run / dst).exists()
        ensure_copy(run, src, dst)
        after = (run / dst).exists()
        made_any = made_any or (not before and after)

    if not made_any:
        print("[i] No new files created. If you don't have the *_hist_test_raw.csv files, run the builder first:")
        print("    python generate_plot_inputs_all.py --run \"out_ecc_scc_mcdo\" --val_split val --test_split test")
    else:
        print("[✓] Aliases ready.")

if __name__ == "__main__":
    main()
