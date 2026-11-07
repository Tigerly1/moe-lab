# Rozdział 4. Analityczny model kosztu warstwy MoE

> Szkic. Liczby z `moelab/roofline.py`, `runs/routing_qwen3_stats.json`, `runs/cmp_*.json` (stan: 2026-11-07).

## 4.1. Notacja

Warstwa MoE z aktywacją SwiGLU:
- `d`: wymiar ukryty modelu.
- `f`: wymiar pośredni jednego eksperta.
- `E`: liczba ekspertów.
- `k`: liczba ekspertów wybieranych dla tokenu (top-k).
- `T`: liczba tokenów w jednym wywołaniu warstwy.
- `b_w`: efektywna liczba bitów na wagę, łącznie z metadanymi kwantyzacji. Dla formatu affine int4 z grupą 64 i 16-bitową skalą oraz biasem `b_w = 4 + 32/64 = 4,5`.

Ekspert `e` liczy

  y_e(x) = W_down,e · ( SiLU(W_gate,e · x) ⊙ (W_up,e · x) ),

a wyjście warstwy to `y = Σ_{j=1..k} w_j · y_{e_j}(x)`.

## 4.2. Praca i ruch danych

- **Operacje:** `FLOP(T) = 6 · d · f · T · k`. Na każdą parę (token, ekspert) przypadają trzy macierze `d×f`, a każde mnożenie z akumulacją to 2 FLOP.
- **Liczba odczytanych ekspertów:** każdy ekspert, którego wybrał choćby jeden z T tokenów, musi zostać odczytany choć raz. Oznaczmy liczbę unikalnych ekspertów przez `U(T)`.
- **Ruch danych:**

  B(T) ≈ U(T) · 3 · d · f · b_w / 8 + O(T · (d + k·f))

  Przy dekodowaniu dominuje pierwszy składnik. Dla Qwen3-30B-A3B przy T = 1 wagi to 21 MB, a aktywacje mniej niż 0,1 MB.

- **Model jednostajny:** jeśli wybory ekspertów są niezależne i jednostajne, to

  U_unif(T) = E · (1 − (1 − k/E)^T)

  (argument typu „zbieracza kuponów”).

- **Intensywność arytmetyczna** względem wag:

  AI = 16 · n̄ / b_w, gdzie n̄ = T·k / U(T) to średnia liczba tokenów na aktywnego eksperta.

  Przy dekodowaniu `n̄ ≈ 1`, więc dla int4 `AI ≈ 3,6` FLOP/B. To o rząd wielkości mniej niż punkt przegięcia (ridge point) obu platform:
  - M4 Pro: około 34 FLOP/B,
  - H100: około 295 FLOP/B.

## 4.3. Dolne ograniczenie czasu i punkt przejścia

- **Dolne ograniczenie czasu:**

  t ≥ max( B(T) / BW, FLOP(T) / P )

  gdzie `BW` to przepustowość pamięci, a `P` to szczytowa wydajność obliczeniowa.

- **Punkt przejścia `T*`** to najmniejsze `T`, przy którym warstwa przestaje być ograniczona pamięcią (int4, model jednostajny):

| Model (d, f, E, k) | Ekspert int4 | t_min(T=1), M4 Pro | T* (M4 Pro) | T* (H100) |
|---|---|---|---|---|
| Qwen3-30B-A3B (2048, 768, 128, 8) | 2,65 MB | 78 µs | 157 | 1537 |
| gpt-oss-20b (2880, 2880, 32, 4) | 14,0 MB | 205 µs | 78 | 746 |
| Mixtral-8x7B (4096, 14336, 8, 2) | 99,1 MB | 726 µs | 39 | 352 |
| DeepSeek-V3 (7168, 2048, 256, 8) | 24,8 MB | 726 µs | 306 | 2834 |

**Wniosek 1.** Na M4 Pro cały praktyczny zakres dekodowania (T ≤ 32) jest ograniczony pamięcią. Optymalizacja polega więc na tym, żeby:
1. czytać każdą potrzebną wagę dokładnie raz (minimalizacja U i duplikatów),
2. osiągać przepustowość bliską BW,
3. nie tracić czasu na stałe koszty, czyli dispatche i synchronizację.

Na H100 granica ta leży o rząd wielkości wyżej. Ta sama architektura warstwy jest tam „bardziej pamięciowa”.

## 4.4. Routing rzeczywisty a model jednostajny

Ślady routingu Qwen3-30B-A3B obejmują 48 warstw i około 5,4 tys. tokenów w trzech domenach: angielski tekst techniczny, kod Python i polski. Pokazują one, że model jednostajny przeszacowuje `U(T)` dla kolejnych tokenów jednej sekwencji:

| T | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| U_unif | 15,5 | 29,1 | 51,6 | 82,4 | 111,8 |
| U, tokeny z różnych sekwencji | 14,9 | 26,2 | 42,6 | 62,5 | 81,9 |
| U, kolejne tokeny | 12,5 | 18,5 | 27,9 | 38,2 | 50,3 |

- **Lokalność czasowa:** sąsiednie tokeny dzielą 43–47% ekspertów. Przy losowym routingu byłoby to 6,2%.
- **Skośność popularności:** 10% najpopularniejszych ekspertów obsługuje 42–59% wyborów.

**Drugi model: gpt-oss-20b (E = 32, k = 4).**

| T | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| U_unif | 7,5 | 13,2 | 21,0 | 28,2 |
| U, kolejne tokeny | 6,1–6,3 | 8,5–9,5 | 12,2–13,4 | 15,7–17,8 |

Sąsiednie tokeny dzielą 41–50% ekspertów. Przy losowym routingu byłoby to 12,5%. Wzorzec jest więc ten sam co w Qwen3, choć architektury są bardzo różne.

**Trzeci model: DeepSeek-V2-Lite (64 ekspertów routowanych i 2 współdzielonych, k = 6).** Lokalność jest tu wyraźnie słabsza:
- sąsiednie tokeny dzielą 18–29% ekspertów, przy 9,4% dla routingu losowego;
- U(8) wynosi 27–30, wobec 34,9 w modelu jednostajnym;
- popularność ekspertów jest niemal równomierna.

Lokalność routingu zależy więc od modelu. Zysk z ponownego użycia wag jest proporcjonalnie mniejszy, co potwierdzają pomiary end-to-end (§6.4c).

**Wniosek 2.** Kernel czytający wagi osobno dla każdej pary (token, slot) przenosi `T·k / U(T)` razy więcej danych niż potrzeba. Dla T = 8 kolejnych tokenów to około 2,3×, a dla T = 16 około 3,4×. Tak wygląda faza weryfikacji w dekodowaniu spekulatywnym. W tym reżimie szybki kernel MoE musi grupować sloty według eksperta, a nie zwiększać przepustowość pojedynczego GEMV.

## 4.5. Weryfikacja modelu (do uzupełnienia)

- [ ] Wykres czasu względem `U(T) · rozmiar eksperta` dla stałego T, na M4 Pro i na GPU NVIDIA. Hipoteza: zależność liniowa, `t ≈ a + b · U`.
- [ ] Pomiar `T*` z eksperymentu dla co najmniej 2 modeli. Porównanie z tabelą z §4.3.
- [ ] Stały koszt `a`, czyli dispatche i synchronizacja. Na M4 Pro jedna para eval+sync w MLX kosztuje około 130 µs po stronie hosta. Koszt po stronie GPU, od kilku do kilkunastu µs na dispatch, wymaga precyzyjnego pomiaru.
