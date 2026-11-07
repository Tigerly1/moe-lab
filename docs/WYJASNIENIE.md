# Co zmieniliśmy i dlaczego to działa

Wyjaśnienie od zera, w duchu Karpathy'ego: zaczynamy od tego, co robi sprzęt, liczymy na kartce, a dopiero potem piszemy kod. Wszystkie liczby dotyczą Apple M4 Pro (48 GB, ~240 GB/s odczytu z pamięci) i są zmierzone (`docs/ACHIEVEMENTS.md`, `runs/`).

## TL;DR

- **Wniosek z rachunku:** dekodowanie modelu MoE na Macu ogranicza przepustowość pamięci, nie obliczenia. Liczy się, ile bajtów wag przeczytasz na krok.
- **Problem stocku:** mlx-lm czyta wagi eksperta osobno dla każdej pary (token, ekspert). Przy wielu zapytaniach naraz ten sam ekspert jest więc czytany wiele razy.
- **Nasza zmiana:** grupujemy tokeny według eksperta i czytamy wagi raz na grupę. Daje to **1,4–1,7× przy 32 zapytaniach naraz** na pięciu popularnych modelach. Pojedyncza rozmowa przyspiesza tylko o 3–17%, bo tam już jesteśmy przy suficie pamięci.
- **Upstream:** kod trafił do mlx-lm jako zgłoszenie [#2044](https://github.com/ml-explore/mlx-lm/pull/2044).

---

## 1. Problem od zera: co się dzieje w warstwie MoE przy generowaniu jednego tokena

Warstwa MoE zawiera E „ekspertów”, czyli małych sieci MLP. Router wybiera dla każdego tokena k z nich (np. Qwen3-30B: E=128, k=8). Każdy ekspert to trzy macierze: gate i up (D×F) oraz down (F×D).

Policzmy Qwen3-30B-A3B (D=2048, F=768), wagi 4-bitowe z grupami 64, czyli ~4,5 bita na wagę:

- jeden ekspert: 3 · 2048 · 768 · 4,5/8 B ≈ **2,65 MB**;
- jeden token w jednej warstwie: 8 ekspertów ≈ **21 MB**;
- 48 warstw: ≈ **1 GB samych ekspertów na token**, a z atencją, embeddingami i lm_head ≈ **1,9 GB na token**;
- przy 240 GB/s daje to sufit ≈ **126 tokenów/s**, choćby GPU liczył nieskończenie szybko.

Obliczeń jest przy tym mało. Mnożenie wektora przez macierz to 2 FLOP-y na wagę, a GPU robi biliony FLOP-ów na sekundę. Pamięć jest więc wąskim gardłem. To pierwsze i najważniejsze spostrzeżenie: **optymalizacja = czytać mniej bajtów albo czytać je szybciej**.

## 2. Co robi stock (mlx-lm) i gdzie traci

mlx-lm liczy ekspertów operacją `gather_qmm`: dla każdej pary (token, wybrany ekspert) mnoży wektor przez wagi tego eksperta. Tracimy w dwóch miejscach.

**a) Jedna rozmowa (T=1 token na krok).**
- Projekcje gate, up i down to osobne wywołania, a każde czyta swój wektor i ma własny narzut uruchomienia.
- Szybka ścieżka MLX (`qmv_fast`) wymaga wymiaru K podzielnego przez 512. Qwen3 ma K=768, gpt-oss ma K=2880, więc obie wpadają na wolniejszą ścieżkę.
- Efekt: stock osiąga ~77% sufitu pamięci.

**b) Wiele zapytań naraz (B tokenów z różnych rozmów w jednym kroku).**
- Każdy z B tokenów wybiera 8 ekspertów, więc przy B=32 jest 256 par.
- Ekspertów jest jednak tylko 128, więc średnio **każdy ekspert jest wybrany ~2 razy**.
- Stock aż do 64 par czyta wagi osobno dla każdej pary, czyli ten sam ekspert przechodzi przez pamięć kilka razy. Przy B=32 stock osiąga ~50% sufitu.

## 3. Co zmieniliśmy

### 3.1. T=1: kernel GEMV, który wykorzystuje pełną szerokość pamięci
Plik: `moelab/variants/mlx/metal_q4_decode_v4.py`.
- **Jeden przebieg dla gate i up:** odczyt wejścia, a potem SiLU(gate)·up od razu w rejestrach.
- **Projekcja down rozdzielona na (ekspert, blok wierszy):** wszystkie 8 ekspertów liczy się równolegle zamiast po kolei, więc w locie jest więcej żądań do pamięci.
- **Sztuczka bitowa:** wejście jest wstępnie przeskalowane przez 1/16^i, więc 4-bitową wagę dekodujemy jednym `and` zamiast przesunięcia i maski. Wynik pozostaje dokładny w fp32.

Wynik: **86–100% sufitu** zamiast 77%.

### 3.2. T>1: grupowanie po ekspercie i mnożenie macierzowe (MMA)
Plik: `moelab/variants/mlx/metal_q4_tiles_v2.py`.
- **Kernel „map”:** układa pary (token, ekspert) w kafle po ≤8 par tego samego eksperta.
- **Jedna grupa wątków SIMD na kafel:** czyta i dekoduje wagi eksperta **raz**, a potem mnoży je przez do 8 tokenów naraz sprzętowym `simdgroup_matrix` 8×8.
- **Kafel z 2 tokenami** idzie ścieżką skalarną, w której jedno dekodowanie wagi obsługuje oba tokeny.

To jest główne źródło zysku przy wielu zapytaniach. Tę samą technikę stosują na NVIDIA vLLM (`fused_moe`) i llama.cpp. Nowe jest przeniesienie jej do MLX/Metal i pomiar zysku na prawdziwym serwerze.

### 3.3. Dyspozytor: kiedy nasz kernel, a kiedy stock
Plik: `moelab/variants/mlx/metal_q4_moe.py`.

Wskaźnik, który decyduje: **obciążenie = T·k/E**, czyli średnia liczba tokenów na eksperta. Liczba tokenów T sama w sobie nie wystarcza.
- Gdy obciążenie jest małe, wygrywa nasz kernel.
- Gdy jest duże (prefill, setki tokenów), wygrywa zoptymalizowany `gather_qmm` z MLX.

Próg zmierzyliśmy siatką pomiarów: 8 dla małych ekspertów, 24 dla dużych. Powyżej progu wołamy ścieżkę **bit-identyczną z mlx-lm**, więc nigdy nie jesteśmy wolniejsi niż stock.

### 3.4. Połączony router
Plik: `moelab/router.py`. Stock wykonuje po kolei softmax po wszystkich ekspertach, `argpartition`, `take` i normalizację, czyli kilka małych operacji. My robimy top-k i wagi w jednym kernelu. Przy 256 ekspertach (Qwen3.6) daje to większość zysku przy jednej rozmowie: 1,03× → 1,11×.

### 3.5. Uniwersalna łatka na `SwitchGLU` (runda 9)
Plik: `moelab/universal.py`.

Każda rodzina modeli ma inny router: softmax, sigmoid z biasem, korekty, eksperci współdzieleni. Zamiast pisać łatkę dla każdej rodziny, podmieniamy tylko wspólny element, czyli `SwitchGLU`, który liczy ekspertów dla par (token, ekspert).
- Każda para trafia do kernela jako osobny wiersz.
- Kernel zwraca wynik dla każdej pary.
- Router i ważenie zostają kodem modelu.

Działa na każdym MoE w mlx-lm. Na Qwen3 kosztuje ~1 punkt procentowy względem łatki dedykowanej.

### 3.6. Drobne, ale ważne
- **GeGLU (Gemma-4):** aktywacja jest parametrem szablonu `ACT` w kernelu.
- **MMA dla każdego wymiaru podzielnego przez 64:** wcześniej wymagaliśmy 128, więc Gemma-4 (F=704) trafiała na wolniejszy wariant. Po poprawce Gemma przy 32 zapytaniach daje **1,25× → 1,70×**.

## 4. Dlaczego wierzymy liczbom

1. **Sufit jako wzorzec.** Liczymy, ile bajtów *musi* zostać przeczytanych (unikalni eksperci × rozmiar), i porównujemy czas z sufitem. Procent sufitu mówi, ile jeszcze da się zyskać. Bez tego łatwo optymalizować coś, co już jest na granicy możliwości.
2. **A/B na przemian, kilka uruchomień procesu, przedziały ufności.** Wahania termiczne i zegara rozkładają się wtedy równo na oba ramiona. Wynik uznajemy dopiero po niezależnym powtórzeniu.
3. **Guard przepustowości.** Przed pomiarem i po nim mierzymy przepustowość pamięci. Jeśli coś innego obciąża GPU (choćby rysowanie ekranu), przebieg oznaczamy jako zakłócony i powtarzamy.
4. **Poprawność względem fp32.** Każda ścieżka jest porównywana z referencją fp32 na tych samych zdekwantyzowanych wagach. Nasz kernel ma o połowę mniejszy błąd niż stock, bo akumuluje w fp32.
5. **Testy dokładne.** Łatkę sprawdzamy, wstawiając w miejsce kernela ścieżkę bit-identyczną ze stockiem. Logity muszą być równe co do bitu. Test mutacyjny potwierdza, że test wykrywa błąd: usuwamy z łatki eksperta współdzielonego i test musi oblać.

## 5. Jak znaleźliśmy lepsze rzeczy

- **Cele z danych, nie z intuicji.** Liczba pobrań repozytoriów mlx-community pokazała, że najpopularniejsze lokalne MoE (Qwen3.6, Gemma-4) nie były obsługiwane. Tam był największy realny wpływ.
- **Mapa luk.** Pomiar procentu sufitu dla różnych T pokazał, gdzie jest zapas, a gdzie już nie:
  - przy T ≤ 8 jesteśmy na 85–100% sufitu, więc nie ma czego szukać;
  - przy T=32 jesteśmy na ~75% (nad tym pracuje agent w tle);
  - prefill i kernele 8-bitowe: najwyżej kilka procent, więc odrzucone po pomiarze, zanim napisaliśmy kod.
- **Czytanie anomalii.** Gemma przyspieszała tylko 1,25×, gdy inne modele 1,5–1,6×. Zamiast to zaakceptować, zmierzyliśmy warstwę osobno. Okazało się, że dyspozytor wysyłał ją na wolniejszy kernel przez zbyt ostry warunek (%128 zamiast %64), a po poprawce zysk wzrósł do 1,70×.
- **Szybko zabijać pomysły.** Każdy pomysł ma kryterium porażki zapisane przed pomiarem (`docs/RESEARCH_BACKLOG.md`).

## 6. Błędy, które złapaliśmy, zanim cokolwiek ogłosiliśmy

| Błąd | Co by wyszło | Jak wykryty |
|---|---|---|
| MLX liczy leniwie; pomiar ewaluował tylko ostatnie wywołanie | 1000+ GB/s, czyli fizycznie niemożliwe | sprawdzenie zdroworozsądkowe względem sufitu |
| Te same wagi w każdym wywołaniu siedziały w cache SLC | ~300 GB/s, ponad sufit | jak wyżej; poprawka: inny routing w każdym wywołaniu |
| A/B w mlx-lm uruchamiane z katalogu gałęzi, więc oba ramiona ładowały ten sam kod | „brak zysku” | identyczne liczby w obu ramionach, wypisanie ścieżki importu |
| Perplexity modelu instruktażowego (Gemma-4) liczona na surowym tekście | PPL ~300, porównanie bez sensu | test na tekście w rozkładzie modelu |
| Tolerancja logitów na małych losowych modelach | fałszywy alarm, bo nawet fp32 różni się o 4–14% | test dokładny z bit-identyczną ścieżką |
| zsh nie rozdziela `$VAR` na argumenty | skrypty po cichu nic nie robiły | wszystkie kolejki teraz w bashu |

## 7. Co dokładnie zmienia zgłoszenie mlx-lm #2044

- **`mlx_lm/models/switch_kernels.py` (nowy plik):**
  - kernele Metal: GEMV dla T=1 (wyjście dla każdego eksperta) i kafle MMA dla T>1;
  - warunki użycia;
  - `mx.custom_function` z gradientem liczonym przez `gather_qmm`, więc trening LoRA działa jak wcześniej.
- **`mlx_lm/models/switch_layers.py`:** 5 linii na początku `SwitchGLU.__call__`. Jeśli szybka ścieżka pasuje, zwraca jej wynik, a jeśli nie, działa dotychczasowy kod. Do tego atrybut `metal_act = 0` na `SwiGLU`.
- **`mlx_lm/models/gemma4_text.py`:** `metal_act = 1` na `GeGLU`.
- **`tests/test_switch_kernels.py`:** zgodność z referencją, gradient i fallbacki.

Szybka ścieżka włącza się tylko przy wagach 4-bitowych z grupą 64, bez biasu, z aktywacjami bf16/fp16, na Metalu i przy obciążeniu poniżej progu. W każdym innym przypadku kod działa dokładnie jak przedtem.

## 8. Wyniki

Benchmark samego mlx-lm (`mlx_lm benchmark`, `main` kontra zgłoszenie), przepustowość generowania:

| Model | 1 rozmowa | 8 naraz | 32 naraz |
|---|---|---|---|
| Gemma-4-26B | 1,05× | 1,29× | **1,71×** |
| Qwen3-30B | 1,08× | 1,20× | **1,56×** |
| DeepSeek-V2-Lite | 1,05× | 1,27× | **1,55×** |
| GLM-4.7-Flash | 1,03× | 1,35× | **1,40×** |
| LFM2.5-8B | 1,06× | 1,33× | **1,38×** |

Prawdziwy `mlx_lm.server` z 32 klientami, z łatką dedykowaną i połączonym routerem:
- Qwen3: 1,38×, mediana odpowiedzi 52 s → 38 s;
- Gemma-4: 1,43×, 53 s → 38 s;
- na Qwen3 jesteśmy 1,48× szybsi od llama.cpp.

## 9. Co dalej

- **T=32:** ~75% sufitu, ostatnia większa luka (agent w tle).
- **NVIDIA (RQ4):** gotowy skrypt `scripts/nvidia_rq4.py`, czeka na wynajem GPU.
- **Drugi Mac (M2):** `scripts/portability_m2.sh`.
- **Poprawka serwera (pusty prompt):** gotowa na forku, do zgłoszenia po #2044.
