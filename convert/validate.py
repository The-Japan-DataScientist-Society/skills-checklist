"""
CSV品質チェックスクリプト

使い方:
  uv run -q python validate.py <csv_dir>

例:
  uv run -q python validate.py output/20260409_185426
  uv run -q python validate.py ..   # ルートのCSVを検証
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema
import polars as pl

# --------------------------------------------------------------------------
# 共通チェック
# --------------------------------------------------------------------------
_no_newline = Check(
    lambda s: ~s.str.contains(r"\n", na=False),
    error="セル内改行あり",
)

# 必須フラグ列（◯ or null のみ許容）: 必須スキル・★必須 系
_maru_value = Check(
    lambda s: s.dropna().isin(["◯"]).all(),
    error="◯ 以外の値が含まれている（期待値: '◯' or null）",
)

# クロスリファレンス列（＊ or null のみ許容）: BZ / DS / DE / VC
_fwast_value = Check(
    lambda s: s.dropna().isin(["＊"]).all(),
    error="＊ 以外の値が含まれている（期待値: '＊' or null）",
)

# スキルレベル列（★ / ★★ / ★★★ のみ許容）
_skill_level_value = Check(
    lambda s: s.dropna().isin(["★", "★★", "★★★"]).all(),
    error="★/★★/★★★ 以外の値が含まれている",
)


# --------------------------------------------------------------------------
# カラム定義ヘルパー
# --------------------------------------------------------------------------
def _id() -> Column:
    """整数IDカラム（No / SubNo）。
    pandasがCSVをint64で読めていることを確認する（float混入・小数値を検出）。"""
    return Column(int, nullable=False)


def _req() -> Column:
    """必須文字列カラム（null不許可）。"""
    return Column(str, nullable=False, checks=_no_newline)


def _maru_col() -> Column:
    """◯ or null のみを許容する必須フラグ列（必須スキル・★必須 系）。"""
    return Column(str, nullable=True, checks=_maru_value)


def _cross() -> Column:
    """他領域との対応マーカー列（BZ / DS / DE / VC）。値は ＊ or null。"""
    return Column(str, nullable=True, checks=_fwast_value)


def _required_skill_cols() -> dict[str, Column]:
    """必須スキル列。foundation / data_* / fusion 共通。値は ◯ or null。"""
    return {"必須スキル": _maru_col()}


def _star_level_cols() -> dict[str, Column]:
    """value_creation のスキルレベル別定義列（★〜★★★）。スキル定義テキストを含む。"""
    return {
        "★（見習い）": _req(),
        "★★（一人前）": _req(),
        "★★★（棟梁）": _req(),
    }


def _star_required_cols() -> dict[str, Column]:
    """value_creation のスキルレベル別必須列（★〜★★★）。値は ◯ or null。"""
    return {
        "★ 必須": _maru_col(),
        "★★ 必須": _maru_col(),
        "★★★ 必須": _maru_col(),
    }


# クロスリファレンス列の正規リスト。ここが唯一の定義元。
_CROSS_NAMES: frozenset[str] = frozenset({"BZ", "DS", "DE", "VC"})


def _cross_cols(*names: str) -> dict[str, Column]:
    """指定したクロスリファレンス列を返す。未知の列名は即エラー（タイポ検出）。"""
    unknown = set(names) - _CROSS_NAMES
    if unknown:
        raise ValueError(f"未知のクロスリファレンス列: {unknown}")
    return {name: _cross() for name in names}


# --------------------------------------------------------------------------
# 共通カラムセット
# --------------------------------------------------------------------------
def _id_cols() -> dict[str, Column]:
    """全CSVに共通するIDカラム (No, SubNo)。"""
    return {
        "No":    _id(),
        "SubNo": _id(),
    }


def _category_cols() -> dict[str, Column]:
    """全5シートに共通するカテゴリカラム。"""
    return {
        "スキルカテゴリ": _req(),
        "サブカテゴリ": _req(),
    }


def _skill_item_cols() -> dict[str, Column]:
    """スキルレベル・チェック項目カラム。foundation + data_* + fusion 共通。
    value_creation は構成が異なるため非対象。"""
    return {
        "スキルレベル": Column(
            str, nullable=False, checks=[_no_newline, _skill_level_value]
        ),
        "チェック項目": _req(),
    }


def _skill_base() -> dict[str, Column]:
    """data_science / data_engineering / fusion 共通の先頭7カラム。
    foundation・value_creation はカラム構成が異なるため非対象。"""
    return {
        **_id_cols(),
        "分類": _req(),
        **_category_cols(),
        **_skill_item_cols(),
    }


# --------------------------------------------------------------------------
# スキーマ定義
# --------------------------------------------------------------------------
SCHEMAS: dict[str, DataFrameSchema] = {
    # foundation: 分類なし → _skill_base() 非対象
    "foundation.csv": DataFrameSchema(
        {
            **_id_cols(),
            **_category_cols(),
            **_skill_item_cols(),
            **_cross_cols("BZ", "DS", "DE"),
            **_required_skill_cols(),
            "旧区分": _req(),
            # Unnamed: 11 は convert.py で削除済み → ここには登場しないはず
        }
    ),
    # value_creation: フェーズ・スキル定義など構成が独自 → _skill_base() 非対象
    "value_creation.csv": DataFrameSchema(
        {
            **_id_cols(),
            "フェーズ": _req(),
            **_category_cols(),
            "スキル定義": _req(),
            **_star_level_cols(),
            **_star_required_cols(),
            **_cross_cols("DS", "DE"),
        }
    ),
    "data_science.csv": DataFrameSchema(
        {
            **_skill_base(),
            **_cross_cols("VC", "DE"),
            **_required_skill_cols(),
        }
    ),
    "data_engineering.csv": DataFrameSchema(
        {
            **_skill_base(),
            **_cross_cols("VC", "DS"),
            **_required_skill_cols(),
        }
    ),
    "fusion.csv": DataFrameSchema(
        {
            **_skill_base(),
            **_required_skill_cols(),
        }
    ),
}


# --------------------------------------------------------------------------
# 診断レポート（エラーではなく情報）
# --------------------------------------------------------------------------
def character_report(df: pd.DataFrame) -> None:
    """通常の日本語・ASCII範囲外の文字を集計して表示する（情報のみ）。"""
    counts: Counter = Counter()
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        for val in df[col].dropna():
            for ch in str(val):
                cp = ord(ch)
                normal = (
                    0x20 <= cp <= 0x7E  # ASCII
                    or 0x3001 <= cp <= 0x30FF  # CJK記号・ひらがな・カタカナ
                    or 0x4E00 <= cp <= 0x9FFF  # CJK統合漢字
                    or 0xFF00 <= cp <= 0xFFEF  # 全角ASCII・半角カタカナ
                    or cp in (0x0A, 0x0D)  # 改行
                ) and cp != 0x3007  # 〇(U+3007) は ◯(U+25EF) と混同しやすいので除外
                if not normal:
                    counts[(cp, ch)] += 1
    if counts:
        print("  [char report]")
        for (cp, ch), n in sorted(counts.items()):
            print(f"    U+{cp:04X} {ch}  x{n}")


# --------------------------------------------------------------------------
# メイン
# --------------------------------------------------------------------------
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    csv_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent
    print(f"Validating CSVs in: {csv_dir.resolve()}\n")

    for fname, schema in SCHEMAS.items():
        path = csv_dir / fname
        print(f"--- {fname} ---")
        if not path.exists():
            print(f"  [SKIP] File not found: {path}")
            print()
            continue

        df = pd.read_csv(path, encoding="utf-8-sig")
        print(f"  pandas : {len(df)} rows x {len(df.columns)} cols")

        # polars でも問題なくロードできるか確認
        df_pl = pl.read_csv(path, encoding="utf8-lossy")
        print(f"  polars : {len(df_pl)} rows x {len(df_pl.columns)} cols")

        # スキーマ外カラムの報告（エラーではなく情報）
        extra_cols = [c for c in df.columns if c not in schema.columns]
        if extra_cols:
            print(f"  [info] スキーマ外カラム: {extra_cols}")

        # No列のEN dash残留チェック（convert.py で落とし損ねた注記行の検出）
        endash_rows = df[df["No"].astype(str).str.strip() == "\u2013"]
        errors: list[str] = []
        if len(endash_rows) > 0:
            errors.append(
                f"EN dash残留 (注記行の取り残し?) 'No': {len(endash_rows)} 行"
            )

        # pandera: 型・null・改行チェック（lazy=True で全エラーを一括収集）
        try:
            schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            for _, row in exc.failure_cases.iterrows():
                col = row.get("column", "?")
                check = row.get("check", "?")
                case = row.get("failure_case", "?")
                errors.append(f"pandera '{col}': {check} — {case!r}")

        # 結果出力
        if errors:
            for e in errors:
                print(f"  [FAIL] {e}")
        else:
            print("  [OK]")

        character_report(df)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
