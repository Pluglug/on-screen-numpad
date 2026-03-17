# On-Screen Numpad Capture Robustness Plan

## 目的

PME の context capture 改修で得た知見を、On-Screen Numpad の direct numeric capture に還元する。
ここでの目的は observation model の導入ではなく、既存の direct 利便性を維持したまま、取得経路の堅牢性とデバッグ性を上げることである。

## 背景

On-Screen Numpad は `User Interface` keymap から直接呼べる点が強みであり、`button_pointer/button_prop` が無い場面でも `context.property` や `copy_data_path_button` を使って数値プロパティへ到達できる。

一方で、PME の capture 改修で実際に問題になった経路のうち、OSN 側にもそのまま潜んでいるものがある。
特に `context.property` の path 分解、clipboard fallback の安全性、Preferences/Theme 系の owner 解決、route ごとの失敗理由可視化は改善余地が大きい。

## 対象スコープ

この計画の対象は次の direct numeric capture 系である。

- `WM_OT_numeric_input.poll()`
- `CalculatorState.is_numeric_property_available()`
- `CalculatorState.detect_property_from_context()`
- `CalculatorState._try_context_property()`
- `CalculatorState._try_copy_data_path_button()`
- `CalculatorState._resolve_path()`

## 改善項目

### 1. `context.property` の path 分解を quoted-aware にする

現状は `split(".")` や `rsplit(".", 1)` ベースの処理があり、次のような owner 名を壊しうる。

- `node_tree.nodes["A.B"]`
- quoted custom property owner
- `[` `]` を含む複雑な segment

最小対応として、quoted segment を壊さず owner path と leaf path を分ける helper を導入する。

期待効果:
- dotted name を含む node / modifier / custom-property owner でも direct capture が落ちにくくなる
- `is_numeric_property_available()` と `_try_context_property()` の判定差を減らせる

### 2. clipboard fallback を安全化する

`copy_data_path_button(full_path=True)` の結果は、成功判定と clipboard 復元を明示した方が安全である。

最小対応:
- `poll()` の確認
- operator result に `FINISHED` が含まれる時だけ採用
- clipboard の退避と復元
- 空文字や無関係な古い clipboard を誤採用しない

期待効果:
- direct fallback が silent wrong result になりにくくなる

### 3. route ごとの provenance を残す

現状は direct capture のログがあっても、「どの route で成功/失敗したか」が追いにくい。

最小対応:
- `button_pointer`
- `context.property`
- `copy_data_path_button`

この 3 経路について、debug ログに route を付ける。
可能なら failure reason も短い code で揃える。

期待効果:
- Preferences / Theme / unusual UI の不具合調査がしやすくなる
- `poll()` と `invoke()` の差分を追いやすくなる

### 4. Preferences / Add-on Preferences / Theme の数値プロパティを強くする

PME 側では次の系統を context path として扱えるようになった。

- `C.preferences.active_section`
- `C.preferences.addons['addon'].preferences.some_value`
- `C.preferences.themes['Default'].image_editor.vertex`

OSN は numeric property 専用なので、Theme 数値との相性が特に良い。

最小対応:
- `context.property` から preferences root を復元できる場合は積極的に使う
- 必要なら nested theme struct の owner path を探索する

期待効果:
- Preferences / Theme の direct numeric edit 体験が広がる

## 優先順位

1. quoted-aware な `context.property` path 分解
2. clipboard fallback の安全化
3. route provenance / failure logging
4. Preferences / Theme 数値対応の拡張

## 非目標

この計画では次を目標にしない。

- PMI のような保存資産モデルの導入
- property / operator / menu の統一 capture model
- 数値以外のプロパティ対応拡張
- UI 全体への汎用 diagnostics framework 導入

## 実装メモ

PME で特に有効だったのは次の方針だった。

- direct convenience は崩さない
- fallback を増やす時ほど、route と失敗理由を記録する
- `context.property` は強いが、path 分解を雑にすると edge case で壊れる
- clipboard fallback は必ず成功判定と restore を入れる

OSN では設計を重くしすぎず、この方針だけを最小限に取り込むのがよい。
