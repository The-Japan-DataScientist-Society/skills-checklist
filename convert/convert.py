import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "input" / "skillcheck_ver6.00.xlsx"
BASE_OUTPUT_DIR = Path(__file__).parent / "output"

SHEETS = {
    "基盤": "foundation.csv",
    "価値創造力": "value_creation.csv",
    "データサイエンス力": "data_science.csv",
    "データエンジニアリング力": "data_engineering.csv",
    "融合": "fusion.csv",
}

# Excel 由来の列名ゆれを正規化するマップ
COLUMN_RENAMES: dict[str, str] = {
    "Sub  No":    "SubNo",       # 基盤: 改行2つ由来のスペース
    "必須 スキル":  "必須スキル",   # データサイエンス力: 改行由来のスペース
    "必須  スキル": "必須スキル",   # 基盤: 改行2つ由来のスペース
    "必須":        "必須スキル",   # 融合: 略称
}


def clean_cell(v):
    if not isinstance(v, str):
        return v
    return v.replace("\n", " ").strip().replace("\u3007", "\u25ef")


def is_blank(v) -> bool:
    return pd.isna(v) or (isinstance(v, str) and v.strip() == "")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_OUTPUT_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading: {INPUT_FILE}")
    print(f"Output:  {output_dir}")
    for sheet_name, csv_name in SHEETS.items():
        df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name, skiprows=2, header=0)
        df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

        # 列名ゆれを正規化
        renamed = {old: new for old, new in COLUMN_RENAMES.items() if old in df.columns}
        if renamed:
            df = df.rename(columns=renamed)
            for old, new in renamed.items():
                print(f"  [RENAME col] {sheet_name}: '{old}' → '{new}'")

        # 変換前に 〇(U+3007) の件数を数えておく
        maru_count = sum(
            str(v).count("\u3007")
            for col in df.columns
            for v in df[col]
            if isinstance(v, str)
        )
        df = df.apply(lambda col: col.map(clean_cell))
        if maru_count > 0:
            print(f"  [CONVERT] {sheet_name}: 〇(U+3007) → ◯(U+25EF) {maru_count} 箇所")

        # Drop unnamed (no header) + all-null/whitespace-only columns
        unnamed_null_cols = [
            c for c in df.columns
            if str(c).startswith("Unnamed:") and df[c].apply(is_blank).all()
        ]
        if unnamed_null_cols:
            print(f"  [DROP cols] {sheet_name}: {unnamed_null_cols}")
            df = df.drop(columns=unnamed_null_cols)

        # No列がEN dash（U+2013）の行は注記行 → 削除
        drop_mask = df["No"].astype(str).str.strip() == "\u2013"
        if drop_mask.any():
            for _, row in df[drop_mask].iterrows():
                print(f"  [DROP] {sheet_name}: No={ascii(row['No'])}, 分類={ascii(str(row.iloc[2])[:30])}")
            df = df[~drop_mask]

        out_path = output_dir / csv_name
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  {sheet_name} -> {csv_name} ({len(df)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
