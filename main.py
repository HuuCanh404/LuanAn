import sys
from pathlib import Path

from models import run_all_models

DEFAULT_CSV = Path(__file__).resolve().parent / "playground" / "cpu_ram_disk_net.csv"


def main():
    """Chay ba mo hinh AI du doan CPU."""
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    print("He thong du doan CPU - system-forecast-do-an")
    print()
    run_all_models(csv_path)


if __name__ == "__main__":
    main()
