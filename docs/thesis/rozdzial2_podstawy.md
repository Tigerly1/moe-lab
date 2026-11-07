# Rozdział 2. Podstawy

> Szkic. Notacja zgodna z §4.1: `d`, `f`, `E`, `k`, `T`, `b_w`, `U(T)`. Numeracja bibliografii według listy na końcu rozdziału 3.

## 2.1. Transformer dekoderowy i wnioskowanie autoregresyjne

**Blok transformera.** Model językowy typu dekoder [1] składa się z `L` bloków. Każdy blok zawiera:
1. normalizację (zwykle RMSNorm) i warstwę uwagi (attention) z pamięcią podręczną kluczy i wartości (KV-cache);
2. normalizację i sieć feed-forward (FFN);
3. połączenia rezydualne wokół obu podwarstw.

Stan przekazywany między blokami to strumień rezydualny `x ∈ R^{T×d}`.

**Dwie fazy inferencji.**
- **Prefill** przetwarza cały prompt naraz. `T` jest duże (setki–tysiące tokenów), mnożenia macierzy są duże i ograniczone obliczeniowo.
- **Dekodowanie** generuje po jednym tokenie na krok, więc `T = 1` na sekwencję. Każda macierz wag jest mnożona przez pojedynczy wektor (GEMV). Na każdy odczytany bajt wag przypada kilka operacji, więc krok jest ograniczony przepustowością pamięci.

**Dekodowanie spekulatywne** [50, 51] częściowo omija to ograniczenie. Mały model pomocniczy (draft) proponuje kilka tokenów, a model docelowy sprawdza je w jednym przejściu (krok weryfikacji). Krok weryfikacji ma `T` od kilku do kilkunastu tokenów jednej sekwencji. Ten sam reżim pojawia się przy serwowaniu z małym batchem, z tą różnicą, że tokeny pochodzą wtedy z różnych sekwencji. Rozróżnienie jest istotne, bo kolejne tokeny jednej sekwencji wybierają podobnych ekspertów (§4.4).

## 2.2. Warstwa Mixture-of-Experts

### 2.2.1. Ekspert SwiGLU

Współczesne modele zastępują klasyczną FFN wariantem z bramkowaniem GLU [14]. Ekspert `e` liczy

  y_e(x) = W_down,e · ( SiLU(W_gate,e · x) ⊙ (W_up,e · x) ),

gdzie `W_gate,e, W_up,e ∈ R^{f×d}`, `W_down,e ∈ R^{d×f}`, a `SiLU(z) = z · σ(z)`. Ekspert ma `3·d·f` wag. Projekcje gate i up czytają to samo `x`, więc naturalnie łączy się je w jednym kernelu razem z SiLU i iloczynem Hadamarda. Niektóre modele modyfikują aktywację, np. gpt-oss stosuje przyciętą wersję SwiGLU i obciążenia (bias) w każdym ekspercie (§6.4b).

### 2.2.2. Router i wybór top-k

Router to liniowa projekcja `W_r ∈ R^{E×d}` dająca logity `r = W_r · x`. Dla każdego tokenu wybieramy `k` ekspertów o największych logitach. Wagi kombinacji powstają przez softmax:
- po wszystkich `E` logitach, a następnie wybór top-k (opcjonalnie z renormalizacją do sumy 1), albo
- tylko po `k` wybranych logitach.

Wyjście warstwy to

  y = Σ_{j=1..k} w_j · y_{e_j}(x).

Warianty routingu:
- **Top-1** (Switch [4]) i **top-2** (GShard [3], Mixtral [8]).
- **Drobnoziarniści eksperci** (fine-grained) z dużym `k`: DeepSeekMoE [9] dzieli ekspertów na mniejsze, przez co `E` i `k` rosną przy stałym koszcie. Qwen3-30B-A3B ma `E = 128`, `k = 8`, `f = 768` [12].
- **Ekspert współdzielony** (shared expert) [9, 10, 11]: jeden lub kilku ekspertów przetwarza każdy token, niezależnie od routera. Z punktu widzenia kernela to zwykła gęsta FFN, wykonywana obok części kierowanej.
- **Bramkowanie sigmoidalne i równoważenie bez straty pomocniczej** w DeepSeek-V3 [11].
- **Expert Choice** [7]: to eksperci wybierają tokeny, a nie odwrotnie.

**Równoważenie obciążenia.** Podczas treningu stosuje się stratę pomocniczą (auxiliary loss), która wyrównuje popularność ekspertów [3, 4], a ST-MoE dodaje router z-loss dla stabilności [6]. Mimo to rozkład wyborów w inferencji pozostaje skośny: w Qwen3-30B-A3B 10% najpopularniejszych ekspertów obsługuje 42–59% wyborów (§4.4).

### 2.2.3. Parametry rozważanych modeli

| Model | d | f | E | k | Ekspert współdzielony |
|---|---|---|---|---|---|
| Qwen3-30B-A3B [12] | 2048 | 768 | 128 | 8 | nie |
| gpt-oss-20b [13] | 2880 | 2880 | 32 | 4 | nie |
| Mixtral-8x7B [8] | 4096 | 14336 | 8 | 2 | nie |
| DeepSeek-V3 [11] | 7168 | 2048 | 256 | 8 | tak (1) |

Rozpiętość jest duża: rozmiar eksperta int4 waha się od 2,65 MB (Qwen3) do 99,1 MB (Mixtral) (§4.3). Kernel dobry dla jednego kształtu nie musi być dobry dla innego.

### 2.2.4. Strategie wykonania warstwy

Dla zadanego routingu (`idx ∈ Z^{T×k}`, `wts ∈ R^{T×k}`) istnieją trzy podstawowe strategie:

| Strategia | Idea | Ruch danych | Zastosowanie |
|---|---|---|---|
| **dense-masked** | każdy ekspert liczy wszystkie tokeny, wyniki nieużywane są maskowane | wszystkie `E` ekspertów | małe `E`, duże `T`; marnuje FLOP-y |
| **gather (per slot)** | dla każdej pary (token, slot) osobny GEMV na wagach eksperta `idx[t,j]` | `T·k` odczytów eksperta (z powtórzeniami) | dekodowanie, T = 1 |
| **sorted grouped GEMM** | permutacja par według eksperta, jeden GEMM na grupę, odwrotna permutacja i kombinacja | `U(T)` odczytów eksperta | duże `T`, prefill |

Sorted grouped GEMM wymaga dodatkowych kroków:
1. sortowania (lub zliczania) par według eksperta i wyrównania grup do rozmiaru kafla (`moe_align_block_size` w vLLM);
2. zebrania (gather) wierszy `x` w kolejności grup;
3. rozproszenia (scatter) wyników i ważonej sumy po slotach (combine).

Przy małym `T` koszt tych kroków dominuje. Strategia gather czyta jednak tego samego eksperta wielokrotnie, jeśli wybrało go kilka tokenów. Kernel z rozdziału 5 łączy obie strategie: grupuje sloty według eksperta w małe kafle bez globalnego sortowania.

## 2.3. Formaty kwantyzacji wag

Kwantyzacja wag zmniejsza `b_w`, a więc ruch danych w reżimie ograniczonym pamięcią. W tej pracy aktywacje pozostają w bf16/fp16 (schemat W4A16). Wybór wartości skwantyzowanych (GPTQ [46], AWQ [47] i inne) jest poza zakresem pracy. Interesuje nas format przechowywania, bo to on wyznacza koszt dekwantyzacji w kernelu.

### 2.3.1. Int4 affine z grupą 64 (MLX)

Wagi dzielimy wzdłuż wymiaru redukcji K na grupy po `g = 64`. Każda grupa ma skalę `s` i przesunięcie `β` w 16 bitach, a waga to

  w = s · q + β,  q ∈ {0, …, 15}.

Efektywna liczba bitów:

  b_w = 4 + 2·16/64 = 4,5.

Iloczyn skalarny grupy można przepisać jako `s · Σ q_i x_i + β · Σ x_i`, więc sumę `Σ x_i` liczymy raz na grupę. Jest to natywny format `mx.quantize` w MLX.

### 2.3.2. MXFP4 (E2M1 ze skalą E8M0)

Format MX (microscaling) [44] przechowuje bloki po 32 elementy ze wspólną skalą:
- element: FP4 E2M1 (1 bit znaku, 2 bity wykładnika, 1 bit mantysy), wartości `±{0; 0,5; 1; 1,5; 2; 3; 4; 6}`;
- skala: E8M0, czyli sam 8-bitowy wykładnik, `2^{e−127}`.

Efektywna liczba bitów:

  b_w = 4 + 8/32 = 4,25.

Format nie ma przesunięcia, więc nie wymaga sumy `Σ x_i`. Używa go gpt-oss dla wag ekspertów [13]. GPU NVIDIA Blackwell wspierają go sprzętowo; na Hopperze i na Apple M4 trzeba go emulować programowo. Skala E8M0 obejmuje zakres znacznie szerszy niż half, co ma konsekwencje dla implementacji (§7.6).

### 2.3.3. FP8 i pozostałe formaty

FP8 występuje w dwóch wariantach [45]: E4M3 (większa precyzja) i E5M2 (większy zakres). Wspierają go tensor cores od Hoppera, a DeepSeek-V3 używa go z blokowym skalowaniem [11]. NVFP4 (FP4 ze skalą FP8 na 16 elementów) jest dostępny tylko na Blackwellu. Apple M4 nie ma sprzętowego wsparcia FP8 ani FP4, więc w tej pracy formaty te pojawiają się tylko jako punkt odniesienia.

| Format | Element | Grupa | Metadane | b_w | Sprzęt |
|---|---|---|---|---|---|
| bf16 | 16 bit | – | – | 16 | wszystkie |
| int8 | 8 bit | 64 | skala + przesunięcie | 8,5 | wszystkie |
| int4 affine g64 | 4 bit | 64 | skala + przesunięcie (fp16) | 4,5 | emulacja |
| MXFP4 | E2M1 | 32 | skala E8M0 | 4,25 | Blackwell; emulacja |
| FP8 (blokowy) | E4M3 | 128 | skala fp32 | ≈ 8 | Hopper+ |

## 2.4. Architektura GPU NVIDIA

**Hierarchia wykonania.**
- GPU składa się z multiprocesorów strumieniowych (SM).
- Wątki są grupowane w **warpy** po 32, wykonujące tę samą instrukcję (SIMT).
- Warpy tworzą bloki wątków (CTA), przydzielane do jednego SM. Wątki bloku współdzielą **pamięć współdzieloną** (shared memory) i mogą się synchronizować.
- SM przełącza się między wieloma aktywnymi warpami, ukrywając opóźnienie pamięci. Liczba aktywnych warpów (obsadzenie, occupancy) zależy od zużycia rejestrów i pamięci współdzielonej.

**Tensor cores.** Jednostki macierzowe wykonują mnożenie małych bloków macierzy (instrukcje `mma` na poziomie warpa; na Hopperze także asynchroniczne `wgmma` na poziomie grupy warpów). Dają one większość szczytowej wydajności GPU dla fp16/bf16/FP8.

**Pamięć.** Hierarchia obejmuje rejestry, shared memory/L1 w każdym SM, wspólny L2 oraz HBM. GPU centrów danych (np. H100 [55]) mają HBM o przepustowości rzędu kilku TB/s. Wysoka wydajność obliczeniowa przesuwa jednak punkt przegięcia do około 295 FLOP/B (bf16, §4.2). W efekcie zakres `T`, w którym warstwa MoE jest ograniczona pamięcią, jest na H100 o rząd wielkości szerszy niż na M4 Pro (tabela w §4.3).

**Koszty stałe.** Każde uruchomienie kernela kosztuje kilka mikrosekund po stronie hosta i GPU. Dla drobnych kerneli dekodowania jest to porównywalne z czasem pracy. Stosuje się CUDA graphs, fuzję kerneli i megakernele [26, 27]. Dekompozycję takich narzutów w inferencji LLM przedstawia [60].

## 2.5. Architektura GPU Apple Silicon

**Hierarchia wykonania (terminologia Metal).**
- GPU składa się z rdzeni GPU; M4 Pro w badanej konfiguracji ma 20 rdzeni.
- Wątki są wykonywane w **grupach SIMD** (SIMD-group) po 32, odpowiednikach warpa. Język Metal udostępnia operacje wewnątrz grupy: `simd_sum`, `simd_shuffle_xor`, `simd_broadcast` [54].
- Grupy SIMD tworzą **grupę wątków** (threadgroup), odpowiednik bloku CUDA. Grupa wątków ma do **32 KB pamięci grupy** (threadgroup memory) [53].

**Operacje macierzowe.** Od generacji Apple7 (M1) Metal udostępnia typ `simdgroup_matrix` z blokami **8×8** dla half, bfloat i float [54]. Jedna grupa SIMD wykonuje mnożenie z akumulacją dwóch takich bloków. Bloki są mniejsze niż w tensor cores NVIDIA, a zmierzona szybkość na M4 Pro to około 6,2 TFLOPS niezależnie od typu (README). Akceleratory neuronowe w GPU pojawiają się dopiero w M5 [36]. M4 nie ma też sprzętowego FP8/FP4.

**Dynamic Caching.** Od M3 GPU przydziela rejestry, pamięć grupy i pamięć podręczną dynamicznie, w czasie wykonania, zamiast statycznie na podstawie najgorszego przypadku [62]. Utrudnia to przewidywanie obsadzenia na podstawie samego kodu kernela.

**Pamięć zunifikowana.** CPU, GPU i Neural Engine współdzielą jedną pamięć LPDDR5X. Bufory nie są kopiowane między hostem a urządzeniem. Między pamięcią a rdzeniami znajduje się wspólna pamięć podręczna systemu (SLC). Wpływ SLC widać w naszych pomiarach: kernel, w którym sloty tego samego eksperta wykonują się współbieżnie, korzysta z ponownego użycia wag przez cache, nawet bez jawnego kafelkowania (`docs/RESULTS.md`, „Mixtral puzzle”).

**M4 Pro w liczbach.**
- Przepustowość pamięci: 273 GB/s w specyfikacji [52]; zmierzony sufit odczytu około 250 GB/s (§6.1).
- Punkt przegięcia: około 34 FLOP/B (§4.2).

**Porównanie pojęć.**

| Pojęcie | NVIDIA (CUDA) | Apple (Metal) |
|---|---|---|
| jednostka obliczeniowa | SM | rdzeń GPU |
| grupa lockstep, 32 wątki | warp | grupa SIMD |
| blok wątków | CTA / thread block | grupa wątków |
| pamięć bloku | shared memory | threadgroup memory (≤ 32 KB) |
| macierze sprzętowe | tensor cores (`mma`, `wgmma`) | `simdgroup_matrix` 8×8 |
| pamięć główna | HBM, oddzielona od hosta | zunifikowana LPDDR, wspólna z CPU |
| cache ostatniego poziomu | L2 | SLC (wspólny z CPU) |
| profilowanie | Nsight Compute | Xcode GPU capture |

## 2.6. Model roofline i intensywność arytmetyczna

Model roofline [15] ogranicza osiągalną wydajność kernela przez:

  P_osiągalne = min( P_szczyt, AI · BW ),

gdzie `AI = FLOP / bajty` to intensywność arytmetyczna. Punkt przegięcia (ridge point) `AI* = P_szczyt / BW` oddziela kernele ograniczone pamięcią (`AI < AI*`) od ograniczonych obliczeniowo. Równoważnie, czas kernela ma dolne ograniczenie

  t ≥ max( B / BW, FLOP / P_szczyt ).

**Zastosowanie w tej pracy.**
- Dla warstwy MoE `AI = 16 · n̄ / b_w`, gdzie `n̄ = T·k / U(T)` (§4.2). Przy dekodowaniu `n̄ ≈ 1`.
- Wynik kernela raportujemy jako **odsetek dolnego ograniczenia**: `t_min / t_zmierzony`, gdzie bajty liczymy dla faktycznie użytych ekspertów. Wartość wyraźnie powyżej 100% oznacza błąd pomiaru lub ponowne użycie danych z cache.
- Jako `BW` i `P_szczyt` podajemy obok wartości katalogowych wartości zmierzone, bo różnica jest istotna (273 vs około 250 GB/s na M4 Pro).

Model roofline zakłada jeden poziom pamięci. Rozszerzenia hierarchiczne uwzględniają kilka poziomów (np. HBM, pamięć hosta, PCIe); MoE-Lightning używa takiego modelu do planowania offloadingu [30]. Styl analizy roofline dla modeli językowych dobrze opisuje [57].

## 2.7. Model programowania Metal i MLX

**Metal.** Programy GPU pisze się w Metal Shading Language, dialekcie C++ [54]. Host:
1. kompiluje kernel do potoku obliczeniowego (compute pipeline);
2. koduje dispatche w buforze poleceń (command buffer) za pomocą enkodera, wiążąc bufory i rozmiar siatki;
3. wysyła bufor do kolejki i, gdy potrzebuje wyniku, czeka na zakończenie.

Siatka jest trójwymiarowa i opisana liczbą wątków oraz rozmiarem grupy wątków.

**MLX** [33] to biblioteka tablicowa Apple dla pamięci zunifikowanej. Kluczowe cechy z punktu widzenia pracy:
- **Leniwa ewaluacja.** Operacje budują graf; obliczenia startują dopiero przy `mx.eval` (lub przy odczycie wartości). Jedno `mx.eval` z synchronizacją kosztuje na M4 Pro około 130 µs po stronie hosta, co przekracza czas warstwy MoE przy dekodowaniu (README). Pomiar czasu musi więc obejmować łańcuch wywołań w jednym grafie (§6.1).
- **Brak kopii host–urządzenie.** Tablice żyją w pamięci zunifikowanej.
- **`mx.compile`** łączy operacje elementowe w jeden kernel. mlx-lm używa go do SwiGLU.
- **Kwantyzacja:** `mx.quantize` (int4/int8 affine, MXFP4) oraz `quantized_matmul`, `gather_qmm` z wyborem ekspertów przez indeksy.
- **`mx.fast.metal_kernel`** [34] pozwala zdefiniować własny kernel z poziomu Pythona. Programista podaje ciało funkcji w MSL, nazwy wejść i wyjść, parametry szablonu, rozmiar siatki i grupy wątków. MLX generuje sygnaturę, kompiluje i buforuje potok, a kernel staje się zwykłym węzłem grafu. Wszystkie kernele z rozdziału 5 są zbudowane w ten sposób.

Warstwa MoE w mlx-lm (`SwitchGLU`) wykonuje trzy wywołania `gather_qmm` (gate, up, down), skompilowane SwiGLU i osobną kombinację. Tokeny są sortowane według eksperta dopiero od 64 slotów. Szczegóły ścieżek MLX omawia §3.4.

## 2.8. CUDA i Triton

**CUDA** to model programowania z jawnym kodem na poziomie wątku, z dostępem do shared memory, instrukcji warpowych i tensor cores. Daje pełną kontrolę, ale wymaga ręcznego zarządzania układem danych. Biblioteki szablonów (CUTLASS) dostarczają gotowych grouped GEMM.

**Triton** [48] to język i kompilator programów na poziomie bloków. Programista opisuje obliczenia jednego programu na kaflach tensorów (`tl.load`, `tl.dot`), a kompilator mapuje je na wątki, shared memory i tensor cores. Konfiguracje kafli (BLOCK_M, BLOCK_N, BLOCK_K, liczba warpów i etapów potoku) dobiera się autotunerem. Kernel `fused_moe` w vLLM [19] jest napisany w Tritonie, z konfiguracjami strojonymi dla każdej kombinacji (E, N, typ, GPU). Triton jest naturalnym odpowiednikiem `mx.fast.metal_kernel` po stronie NVIDIA, dlatego wybraliśmy go do badania przenośności (RQ4).

## 2.9. JAX, XLA i Pallas

**JAX** [61] śledzi funkcje Pythona do pośredniej reprezentacji i kompiluje je przez **XLA** (`jax.jit`), który sam łączy operacje i wybiera kernele. Dla MoE JAX udostępnia `jax.lax.ragged_dot`, czyli grouped GEMM ze zmiennymi rozmiarami grup.

**Pallas** [56] to rozszerzenie JAX do pisania własnych kerneli na poziomie bloków, podobne do Tritona, z backendem Mosaic dla TPU i GPU. Grouped GEMM dla MoE w Pallas to Megablox, używany w MaxText [59]; biblioteka Tokamax [58] udostępnia `ragged_dot` z autotunerem. Pallas nie ma backendu dla Apple GPU, więc port JAX w tej pracy służy porównaniu z TPU i NVIDIA, a nie z Metal.

## 2.10. Podsumowanie

Trzy fakty z tego rozdziału wyznaczają przestrzeń projektową kerneli:
1. W reżimie dekodowania warstwa MoE jest ograniczona pamięcią na obu platformach, więc liczy się `U(T) · b_w`, a nie FLOP-y.
2. Apple GPU ma mniejsze bloki macierzowe, mniejszą pamięć grupy wątków i pamięć zunifikowaną ze wspólnym SLC. Gotowe rozwiązania z NVIDIA nie przenoszą się więc wprost.
3. Wszystkie trzy ekosystemy (Metal/MLX, Triton, Pallas) pozwalają pisać kernele na poziomie bloków lub grup SIMD z Pythona, co umożliwia wspólny kontrakt wariantu (§5.1).
