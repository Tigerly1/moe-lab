# Rozdział 1. Wstęp

> Szkic. Liczby wyników pochodzą z `README.md` (sekcja „Headline result”) i `docs/RESULTS.md` (stan: 2026-11-07). Numeracja bibliografii według listy na końcu rozdziału 3.

## 1.1. Motywacja

**Mixture-of-Experts jako standard dużych modeli językowych.** Warstwa Mixture-of-Experts (MoE) zastępuje gęstą sieć feed-forward transformera [1] zbiorem `E` ekspertów, z których router wybiera dla każdego tokenu tylko `k` [2, 3, 4]. Liczba parametrów rośnie wtedy z `E`, a koszt obliczeń na token tylko z `k`. Ten kompromis przyjęła większość otwartych modeli z czołówki:
- Mixtral-8x7B: 8 ekspertów, top-2 [8];
- DeepSeek-V3: 256 ekspertów kierowanych i ekspert współdzielony, top-8 [11];
- Qwen3-30B-A3B: 128 ekspertów, top-8 [12];
- gpt-oss-20b: 32 ekspertów, top-4, wagi ekspertów w formacie MXFP4 [13].

**Dekodowanie jest ograniczone pamięcią.** Generowanie tekstu przebiega token po tokenie. W każdym kroku warstwa MoE musi odczytać wagi wszystkich wybranych ekspertów, a wykonuje na nich niewiele operacji. Dla dekodowania pojedynczej sekwencji intensywność arytmetyczna wynosi około 3,6 FLOP/B przy wagach int4 (§4.2). To o rząd wielkości poniżej punktu przegięcia modelu roofline [15] zarówno dla Apple M4 Pro (około 34 FLOP/B), jak i dla NVIDIA H100 (około 295 FLOP/B). Czas warstwy wyznacza więc przepustowość pamięci, a nie moc obliczeniowa. Dobry kernel MoE dla dekodowania to kernel, który:
1. czyta każdą potrzebną wagę dokładnie raz,
2. czyta ją z prędkością bliską sprzętowemu limitowi,
3. nie traci czasu na stałe koszty uruchomień.

**Dwie platformy, dwie hierarchie pamięci.**
- **NVIDIA:** GPU centrów danych mają pamięć HBM o bardzo dużej przepustowości, ale ograniczonej pojemności. Duże modele MoE wymagają tam wielu GPU albo przenoszenia ekspertów do pamięci hosta [29, 30, 31].
- **Apple Silicon:** układy serii M mają pamięć zunifikowaną (unified memory), wspólną dla CPU i GPU. Przepustowość jest mniejsza (273 GB/s w specyfikacji M4 Pro [52]), ale pojemność pozwala trzymać cały skwantyzowany model MoE w pamięci dostępnej dla GPU. Dlatego Apple Silicon stał się popularną platformą lokalnej inferencji (MLX [33], llama.cpp [35]).

Modele MoE pasują do tego profilu wyjątkowo dobrze: mają dużo parametrów (potrzebna pojemność), ale aktywują mało na token (wystarczy umiarkowana przepustowość). Mimo to ścieżki MoE w bibliotekach dla Apple Silicon są znacznie słabiej zoptymalizowane niż ich odpowiedniki dla NVIDIA (rozdział 3). Literatura kerneli MoE dotyczy niemal wyłącznie GPU NVIDIA i AMD.

## 1.2. Problem badawczy

Praca dotyczy wykonania warstwy MoE w inferencji, przy zadanym routingu, dla małych i średnich liczb tokenów `T` na wywołanie. Ten zakres obejmuje:
- zwykłe dekodowanie (T = 1);
- krok weryfikacji w dekodowaniu spekulatywnym [50, 51], gdzie model sprawdza naraz kilka do kilkunastu tokenów zaproponowanych przez model pomocniczy;
- serwowanie z małym batchem.

Pytanie centralne brzmi: **jak zaprojektować kernele warstwy MoE, które dla każdego reżimu T zbliżają się do dolnego ograniczenia wynikającego z modelu roofline, i w jakim stopniu te decyzje projektowe przenoszą się między GPU Apple a GPU NVIDIA?**

Dwie trudności odróżniają ten problem od zwykłego mnożenia macierzy:
1. **Rzadkość zależna od danych.** Zbiór ekspertów do odczytania zależy od routingu, znanego dopiero w czasie wykonania. Liczba unikalnych ekspertów `U(T)` decyduje o ruchu danych (rozdział 4).
2. **Kwantyzacja.** Wagi ekspertów są przechowywane w formatach 4-bitowych (int4 affine, MXFP4), więc kernel musi łączyć dekwantyzację z mnożeniem bez utraty przepustowości.

## 1.3. Pytania badawcze

| | Pytanie | Hipoteza / miara |
|---|---|---|
| **RQ1** | **Punkt przejścia reżimów.** Która strategia wygrywa dla T = 1…4096: dense-masked, gather-GEMV (jedna grupa SIMD/warp na wiersz) czy posortowany grouped GEMM? | Punkt przejścia podąża za punktem przegięcia architektury; model analityczny przewiduje go z dokładnością około 20% na M4 Pro i na NVIDIA. |
| **RQ2** | **Kwantyzowane dekodowanie na Metal.** Jak blisko sufitu przepustowości może zejść połączony (fused) kernel int4 MoE GEMV? | Odsetek dolnego ograniczenia roofline; porównanie z mlx-lm oraz z Marlin-MoE na NVIDIA. |
| **RQ3** | **Wartość fuzji.** Co daje każda fuzja (gate+up+SwiGLU, down+combine, wyrównanie po stronie GPU) w każdym reżimie i na każdej architekturze? | Ablacja wariantów, przyspieszenie z przedziałem ufności. |
| **RQ4** | **Przenośność.** Czy konfiguracje kafli i struktura kernela przenoszą się między Metal a Triton? Czy przenoszą się warianty znalezione automatycznie? | Porównanie tych samych decyzji projektowych na obu platformach i dla różnych formatów kwantyzacji. |
| **RQ5** | **Automatyczne przeszukiwanie** (opcjonalne). Siatka vs optymalizacja bayesowska vs mutator LLM na tym samym ewaluatorze. | fast_p [39], liczba ewaluacji, liczba wykrytych prób oszukania metryki. |

## 1.4. Wkład pracy

1. **Model analityczny warstwy MoE z rzeczywistym routingiem** (rozdział 4). Model wiąże ruch danych z liczbą unikalnych ekspertów `U(T)` i wyznacza reżimy pracy oraz punkt przejścia `T*` dla obu platform. Ślady routingu Qwen3-30B-A3B pokazują, że sąsiednie tokeny dzielą około 45% ekspertów (43–47%, około 7× więcej niż przy losowym routingu), a rzeczywiste `U(T)` jest mniej więcej o połowę mniejsze od przewidywań modelu jednostajnego.

2. **Środowisko badawcze `moe-lab`** (rozdział 5). Wspólny kontrakt wariantu dla MLX/Metal, PyTorch/Triton i JAX/Pallas. Bramka poprawności względem referencji fp32/fp64, izolacja procesów, przeplatane pomiary A/B z bootstrapem hierarchicznym, pomiar czasu jako nachylenie względem długości łańcucha oraz laboratorium interaktywne do edycji kerneli.

3. **Kernele Metal dla int4 affine i MXFP4** z dyspozytorem reżimów (rozdział 5). Przy T = 1 kernel GEMV z pełnym wykorzystaniem linii SIMD; przy T > 1 kernel grupujący sloty według eksperta, który czyta wagi eksperta raz na kafel i liczy na `simdgroup_matrix` 8×8. Wyniki na M4 Pro (rozdział 6):
   - **Pełny model, mlx-lm:**

     | Model | Dekodowanie | Weryfikacja, 4 tokeny | Weryfikacja, 16 tokenów |
     |---|---|---|---|
     | Qwen3-30B-A3B-4bit | 91,6 → 99,8 tok/s (+9,0%) | +23% | +45% |
     | gpt-oss-20b-MXFP4 | 75,5 → 81,5 tok/s (+7,9%) | +27% | +70% |

     Perplexity bez zmian; kernele są co najmniej tak bliskie referencji fp32 jak ścieżka mlx-lm.
   - **Warstwa MoE względem mlx-lm (Qwen3, ślady rzeczywiste):** int4 1,20× przy T = 1 i 2,46× przy T = 32; MXFP4 1,26× i 2,37×.
   - **Kształt w pełni spoza strojenia (Qwen3-235B, routing jednostajny):** int4 1,06–1,26×, MXFP4 1,07–1,23×.
   - **T = 1 na ścianie pamięci:** przy niepowtarzającym się routingu kernel czyta z prędkością 94% (Qwen3), 100% (gpt-oss) i około 103% (Mixtral) z 250 GB/s; mlx-lm osiąga 77% na Qwen3.

4. **Wyjaśnienie mechanizmów zysku** (rozdział 6).
   - Przy T = 1 około połowa przewagi wynika z warunku K % 512 = 0 szybkiej ścieżki `qmv` w MLX, którego nie spełniają Qwen3 (K = 768) ani gpt-oss (K = 2880). Reszta to fuzja i pełne wykorzystanie linii. Wynik jest bezpośrednio związany z otwartym zgłoszeniem MLX #4509.
   - Przy T > 1 zysk pochodzi z lokalności routingu: kernel czyta wagi każdego eksperta raz na kafel. Przy T ≥ 8 staje się ograniczony obliczeniowo (około 80% zmierzonej szybkości MMA, 6,2 TFLOPS).

5. **Metodyka poszukiwań wspieranych przez agenta LLM** (autoresearch, §5.6, RQ5), wraz z opisem zabezpieczeń przed oszukiwaniem metryki i błędów samego agenta.

6. **Udokumentowane korekty** (rozdział 7). Wszystkie liczby przeszły dwa niezależne audyty. Poprawiliśmy m.in.:
   - nieuczciwą pierwszą bazę (przyspieszenie 1,25× zamiast 1,17×);
   - zanikanie aktywacji w łańcuchu pomiarowym;
   - zafałszowanie tok/s w dekodowaniu spekulatywnym przez odsetek akceptacji;
   - dwa błędy przepełnienia skali MXFP4, ujawnione dopiero przez rzeczywisty checkpoint gpt-oss;
   - zbyt wąskie przedziały ufności dla routingu ze śladów;
   - zawyżone etykiety „held-out” i odsetki przepustowości.

**Ograniczenia stanu obecnego.**
- Pomiary dotyczą jednego układu (M4 Pro). Inne generacje Apple nie były testowane.
- Kernele Triton dla NVIDIA nie zostały jeszcze uruchomione na GPU. Port JAX/Pallas jest sprawdzony tylko w trybie interpretera na CPU. RQ4 jest więc rozpoczęte, a RQ1 częściowe.
- Swobodne dekodowanie spekulatywne z modelem pomocniczym 0,6B jest na tej konfiguracji wolniejsze niż zwykłe dekodowanie (72–77 wobec 91,5 tok/s). Zysk w kroku weryfikacji dotyczy tylko tego kroku.

## 1.5. Struktura pracy

- **Rozdział 2** wprowadza podstawy: transformer i warstwę MoE, formaty kwantyzacji, architektury GPU NVIDIA i Apple, model roofline oraz modele programowania (Metal/MLX, CUDA/Triton, JAX/Pallas).
- **Rozdział 3** omawia prace pokrewne: architektury MoE, systemy i kernele MoE, inferencję na sprzęcie konsumenckim, ścieżki MoE na Apple Silicon i automatyczną optymalizację kerneli.
- **Rozdział 4** buduje analityczny model kosztu warstwy MoE z uwzględnieniem rzeczywistego routingu.
- **Rozdział 5** opisuje środowisko badawcze i kolejne wersje kerneli.
- **Rozdział 6** przedstawia ewaluację, z sekcją dla każdego pytania badawczego.
- **Rozdział 7** omawia zagrożenia dla trafności wyników.
- **Rozdział 8** zawiera wnioski. Dodatek opisuje odtwarzanie wyników.
