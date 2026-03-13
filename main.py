from pathlib import Path

from models import run_all_models

CSV_PATH = Path(__file__).resolve().parent / "playground" / "cpu_ram_disk_net.csv"


def main():
    """Chay ba mo hinh AI du doan CPU."""
    print("He thong du doan CPU - system-forecast-do-an")
    print()
    run_all_models(CSV_PATH)


if __name__ == "__main__":
    main()
