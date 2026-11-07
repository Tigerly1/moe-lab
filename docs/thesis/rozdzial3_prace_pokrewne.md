# Rozdział 3. Prace pokrewne

> Szkic. Identyfikatory arXiv w bibliografii sprawdzono 2026-11-07 na arxiv.org (tytuł i pierwszy autor). Znak ⚠ oznacza informację, której nie udało się potwierdzić w źródle pierwotnym.

## 3.1. Architektury Mixture-of-Experts

**Początki.** Shazeer i in. [2] wprowadzili rzadko bramkowaną warstwę MoE między warstwami LSTM. Router z szumem (noisy top-k gating) wybierał kilku ekspertów spośród tysięcy, a strata pomocnicza równoważyła ich obciążenie. Praca pokazała, że pojemność modelu można zwiększać o rzędy wielkości przy prawie stałym koszcie obliczeń.

**MoE w transformerach.**
- **GShard** [3] zastąpił co drugą FFN transformera warstwą MoE z routingiem top-2, pojemnością ekspertów (capacity factor) i automatycznym shardingiem ekspertów między urządzenia.
- **Switch Transformer** [4] uprościł routing do top-1 i pokazał skalowanie do modeli o bilionie parametrów.
- **GLaM** [5] zastosował MoE w modelu językowym typu dekoder i osiągnął jakość porównywalną z gęstymi modelami przy znacznie mniejszym koszcie energii.
- **ST-MoE** [6] zajął się stabilnością treningu (router z-loss) i przenoszeniem modeli rzadkich przez fine-tuning.
- **Expert Choice** [7] odwrócił kierunek wyboru: każdy ekspert wybiera stałą liczbę tokenów. Daje to idealne równoważenie, ale wybór zależy od całego batcha, co komplikuje dekodowanie autoregresyjne.

**Otwarte modele MoE (2024–2025).**
- **Mixtral-8x7B** [8]: 8 dużych ekspertów, top-2. Spopularyzował MoE w otwartych modelach i lokalnej inferencji.
- **DeepSeekMoE** [9]: drobnoziarniści eksperci (podział eksperta na mniejsze przy większym `k`) i eksperci współdzieleni, przechwytujący wiedzę wspólną.
- **DeepSeek-V2** [10] i **DeepSeek-V3** [11]: architektura DeepSeekMoE w dużej skali. V3 ma 256 ekspertów kierowanych, 1 współdzielonego, top-8, bramkowanie sigmoidalne, równoważenie bez straty pomocniczej i trening w FP8.
- **Qwen3-MoE** [12]: 128 ekspertów, top-8, bez ekspertów współdzielonych. Qwen3-30B-A3B jest głównym modelem tej pracy.
- **gpt-oss** [13]: 32 (20b) lub 128 (120b) ekspertów, top-4, wagi ekspertów w MXFP4, przycięta aktywacja SwiGLU z obciążeniami.

**Trend.** Kierunek zmian jest wyraźny: więcej mniejszych ekspertów (rosnące `E` i `k`, malejące `f`) oraz niższa precyzja wag. Oba trendy pogarszają sytuację kerneli: mniejsze macierze to więcej narzutu na uruchomienie i gorsze wykorzystanie linii, a niższa precyzja przenosi część kosztu na dekwantyzację. Qwen3-30B-A3B (`f = 768`) jest dobrym przykładem modelu, dla którego standardowe ścieżki MLX zawodzą (§6.2).

## 3.2. Systemy i kernele MoE na GPU NVIDIA

### 3.2.1. Systemy treningowe

- **DeepSpeed-MoE** [18] łączy architekturę PR-MoE z systemem inferencji: równoległość ekspertów, zoptymalizowane kernele i komunikację all-to-all.
- **Tutel** [17] adaptacyjnie przełącza strategię równoległości i potokowania w czasie wykonania, bo obciążenie ekspertów zmienia się dynamicznie.
- **MegaBlocks** [16] formułuje MoE jako mnożenie macierzy blokowo-rzadkich. Dzięki temu nie trzeba odrzucać tokenów ani dopełniać grup do stałej pojemności (dropless MoE).
- **ScatterMoE** [23] implementuje MoE w Tritonie bez dopełniania i bez dodatkowych kopii, łącząc gather/scatter z GEMM.
- **SonicMoE** [24] optymalizuje MoE pod kątem ruchu danych i rozmiaru kafli (IO- and tile-aware) na nowych GPU NVIDIA.

Systemy te optymalizują głównie trening i prefill, czyli reżim dużych `T`.

### 3.2.2. Kernele inferencji

- **vLLM `fused_moe`** [19]. Kernel w Tritonie: `moe_align_block_size` grupuje pary (token, slot) według eksperta i wyrównuje grupy do BLOCK_M, kernel GEMM wykonuje niejawny gather wierszy `x`, a waga routingu jest mnożona w epilogu. Konfiguracje kafli strojono dla każdej kombinacji (E, N, typ, GPU) i zapisano jako pliki JSON.
- **Marlin** [20]. Kernel W4A16 osiągający przyspieszenie bliskie idealnemu (wynikającemu ze zmniejszenia bajtów) dla małych batchy. W vLLM rozszerzono go o wariant MoE (Marlin-MoE), który jest naszym planowanym punktem odniesienia dla RQ2 na NVIDIA.
- **DeepGEMM** [21]. Biblioteka DeepSeek z FP8 GEMM, w tym grouped GEMM w dwóch układach: ciągłym (contiguous, prefill) i maskowanym (masked, dekodowanie).
- **TensorRT-LLM** [22] opisuje wykonanie MoE jako gęstego GEMM na B200. Wariant dense-masked wygrywa z grouped GEMM w zakresie T ≈ 64–208, mimo że około 97% FLOP-ów jest marnowanych. To bezpośredni przykład punktu przejścia reżimów, który badamy w RQ1.
- **FlashInfer** [25]. Silnik attention dla serwowania LLM; repozytorium zawiera też kernele MoE oparte na CUTLASS (`cutlass_fused_moe`), z niedeterministyczną kombinacją wyników przez operacje atomowe.

### 3.2.3. Megakernele i przenośność

- **FlashMoE** [26] wykonuje całą rozproszoną warstwę MoE, łącznie z komunikacją, w jednym kernelu.
- **MonoMoE** [27] to megakernel dla kwantyzowanego dekodowania MoE, eliminujący narzut wielu uruchomień.
- **TritonMoE** [28] proponuje przenośny dispatch MoE w Tritonie, bez zależności od CUDA. To najbliższa praca dotycząca przenośności między architekturami, ale obejmuje tylko GPU dyskretne (NVIDIA i AMD ⚠ szczegóły platform niezweryfikowane).
- **TaxBreak** [60] rozkłada narzuty inferencji LLM (uruchomienia kerneli, host) na składowe, co uzasadnia fuzję w reżimie dekodowania.

**Podsumowanie.**

| Praca | Reżim | Precyzja | Platforma | Główna idea |
|---|---|---|---|---|
| MegaBlocks [16] | trening | bf16 | NVIDIA | blokowo-rzadki GEMM, dropless |
| ScatterMoE [23] | trening/prefill | bf16 | NVIDIA (Triton) | gather/scatter w GEMM |
| SonicMoE [24] | trening | bf16 | NVIDIA | IO- i tile-aware |
| vLLM fused_moe [19] | inferencja | bf16/FP8/int | NVIDIA, AMD (Triton) | align + niejawny gather |
| Marlin(-MoE) [20] | dekodowanie | W4A16 | NVIDIA | dekwantyzacja w potoku, pełna przepustowość |
| DeepGEMM [21] | prefill/dekodowanie | FP8 | NVIDIA Hopper | grouped GEMM contiguous/masked |
| TRT-LLM [22] | średnie T | FP4/FP8 | NVIDIA Blackwell | MoE jako gęsty GEMM |
| MonoMoE [27] | dekodowanie | kwantyzowana | NVIDIA | megakernel |
| TritonMoE [28] | inferencja | – | wieloplatformowy Triton | przenośny dispatch |

## 3.3. Offloading i sprzęt konsumencki

Gdy model MoE nie mieści się w pamięci GPU, eksperci są przechowywani w pamięci hosta lub na dysku:
- **Fiddler** [29] wykonuje ekspertów nieobecnych w GPU na CPU zamiast przesyłać ich wagi przez PCIe.
- **MoE-Lightning** [30] potokuje obliczenia CPU, GPU i transfery I/O, a politykę wybiera na podstawie hierarchicznego modelu roofline.
- **MoE-Infinity** [31] buforuje ekspertów na komputerach osobistych, przewidując aktywacje na podstawie śladów routingu.
- **Pre-gated MoE** [32] modyfikuje router tak, by wybierał ekspertów dla następnej warstwy, co pozwala pobierać ich z wyprzedzeniem.

Prace te wykorzystują lokalność i skośność routingu do decyzji o przenoszeniu ekspertów. Nasza praca używa tej samej własności na innym poziomie hierarchii: do współdzielenia odczytów wag między tokenami wewnątrz kernela (§4.4). Pamięć zunifikowana Apple Silicon usuwa problem PCIe, ale nie problem przepustowości.

## 3.4. MoE na Apple Silicon

**MLX i mlx-lm** [33]. Warstwa `SwitchGLU` wykonuje trzy wywołania `gather_qmm`. Z analizy kodu źródłowego MLX:
- tokeny są sortowane według eksperta dopiero od 64 slotów (`indices.size ≥ 64`);
- `gather_qmm_rhs` (wariant korzystający z posortowania) jest wybierany przy `M == 1 && B ≥ 16 && sorted && B/E ≥ 4`, z komentarzem `// TODO: Tune 16 and 4`;
- poniżej progu wektorowego używany jest `gather_qmv`;
- szybka ścieżka `qmv` dla 4 bitów wymaga K % 512 = 0 (§6.2).

Zgłoszenie MLX #4509 dokumentuje, że `gather_qmm` przy M = 1 czyta wagi ekspertów z mniej więcej połową przepustowości osiąganej przy M = 4. Węższą poprawkę (PR #4598, ścieżka dla K % 256) odrzucono z powodu złożoności dyspozycji. Nasze wyniki dla RQ2 dostarczają ablacji tego zjawiska.

**llama.cpp** [35]. Backend Metal stosuje dla MoE:
- prepass `map0`, budujący listy kafli dla każdego eksperta;
- `kernel_mul_mm_id` z kaflami NR0 = 64, NR1 = 32 na `simdgroup_matrix` 8×8 dla większych batchy;
- `mul_mv_id` dla dekodowania.

Akceleratory tensorowe są używane tylko na M5 i nowszych. Struktura jest zbliżona do naszego kernela kafelkowego, ale progi i rozmiary kafli są stałe, a nie dobierane według reżimu i lokalności routingu.

**Akceleratory poza GPU.**
- **BaseRT** [36] wykorzystuje akceleratory neuronowe w GPU Apple M5 do inferencji LLM. M4 Pro ich nie ma, co oddziela naszą pracę od BaseRT.
- **NPUMoE** [37] uruchamia modele MoE na Apple Neural Engine. Statyczne kształty grafów NPU są sprzeczne z dynamicznym routingiem, więc praca stosuje kalibrację pojemności ekspertów offline i fallback na CPU/GPU.

Obie prace dotyczą innych jednostek niż GPU M1–M4, na którym działają MLX i llama.cpp.

## 3.5. Automatyczna optymalizacja kerneli

**Benchmarki i metryki.**
- **KernelBench** [39] mierzy zdolność LLM do pisania kerneli GPU. Metryka fast_p to odsetek zadań, w których kernel jest poprawny i co najmniej p razy szybszy od referencji.
- **CUDA-L1** [43] stosuje uczenie ze wzmocnieniem do optymalizacji CUDA i dokumentuje przypadki oszukiwania metryki (reward hacking), m.in. przez uruchamianie obliczeń na osobnym strumieniu, którego pomiar czasu nie obejmuje. Z tej pracy pochodzi nasza zasada synchronizacji całego urządzenia przed zdarzeniem końcowym.

**Systemy poszukiwań.**
- **AlphaEvolve** [38] łączy LLM z algorytmem ewolucyjnym i automatycznym ewaluatorem. Wśród wyników są przyspieszenie o 23% kluczowego kernela mnożenia macierzy w Gemini (Pallas) i do 32,5% kernela FlashAttention [49].
- **autoresearch** [40] Karpathy'ego to minimalny protokół: człowiek pisze `program.md`, agent edytuje jeden plik, ewaluator jest tylko do odczytu, a każdy eksperyment to commit zachowywany albo cofany. Nasza pętla (§5.6) jest jego adaptacją do kerneli Metal.
- **AutoKernel** [41] profiluje model PyTorch, szereguje wąskie gardła według prawa Amdahla i iteracyjnie poprawia kernele Triton/CUDA w pętli agenta z wieloetapową bramką poprawności.
- **K-Search** [42] oddziela planowanie algorytmiczne od implementacji i zastępuje statyczne heurystyki ewolucyjne współewoluującym modelem świata. Ewaluacja obejmuje kernele z FlashInfer, w tym MoE. ⚠ Informacja, że praca wymienia „fused MoE na MLX” jako przyszły kierunek, pochodzi z wyszukiwania i nie została potwierdzona w tekście.

Wszystkie te systemy działają na GPU NVIDIA (CUDA, Triton) lub TPU (Pallas). Nie znaleźliśmy pracy stosującej poszukiwania wspierane przez LLM do kerneli Metal.

## 3.6. Pozycjonowanie pracy

| Cecha | Kernele NVIDIA [16–28] | Offloading [29–32] | Apple (MLX, llama.cpp, BaseRT, NPUMoE) | Ta praca |
|---|---|---|---|---|
| GPU Apple M1–M4 | – | – | MLX, llama.cpp (bez analizy) | tak |
| porównanie z NVIDIA przy wspólnej metodzie | – | – | – | tak (RQ1, RQ4; w toku) |
| model roofline per reżim T | częściowo (TRT-LLM) | hierarchiczny [30] | – | tak (rozdział 4) |
| rzeczywista lokalność routingu w kernelu | – | do offloadingu | – | tak (§4.4, §6.3) |
| formaty int4 affine i MXFP4 | Marlin; MXFP4 natywnie na Blackwell | – | MLX (bez fuzji) | tak, emulacja z fuzją |
| poszukiwania wspierane przez LLM | AlphaEvolve, KernelBench, AutoKernel, K-Search | – | – | tak (RQ5) |

**Luka.** O ile nam wiadomo, nie ma pracy, która badałaby kernele warstwy MoE na GPU Apple i na GPU NVIDIA przy wspólnej metodzie: tym samym kontrakcie wariantu, tej samej bramce poprawności i tym samym modelu roofline z liczbą unikalnych ekspertów `U(T)` mierzoną na rzeczywistym routingu. Najbliższe prace albo dotyczą jednej rodziny GPU dyskretnych (TritonMoE, MonoMoE), albo innych jednostek Apple (BaseRT, NPUMoE). Lokalność routingu była dotąd wykorzystywana do decyzji o przenoszeniu ekspertów, a nie do projektowania kernela dla kroku weryfikacji. Ta praca wypełnia tę lukę dla Apple M4 Pro i przygotowuje tę samą metodę dla NVIDIA i TPU.

## 3.7. Lokalność routingu i obsługa równoległych zapytań (uzupełnienie po rundzie 8)

- **Lokalność czasowa routingu jest znana.**
  - Mixtral [8] raportuje, że ekspert pierwszego wyboru powtarza się w 24–28% kolejnych tokenów, przy 12,5% dla routingu losowego.
  - Liang i in. [63] mierzą spójność routingu w segmentach tokenów dla 20 modeli MoE, w tym Qwen3 i DeepSeek-V2-Lite (bez gpt-oss). Pokazują, że eksperci wyspecjalizowani dziedzinowo zwiększają tę spójność, a eksperci współdzieleni ją obniżają.
  - Qiu i in. [64] pokazują, że globalne równoważenie obciążenia, stosowane w Qwen3-MoE, wzmacnia specjalizację dziedzinową ekspertów.
- **„Super Experts” [65]** to inne zjawisko: kilku ekspertów niezależnych od danych, z ekstremalnymi aktywacjami. Nasz „gorący ekspert” (§4.4) zależy od dokumentu.
- **Grupowanie tokenów według eksperta jest standardem na GPU NVIDIA:** vLLM `fused_moe` [19] i llama.cpp. W llama.cpp wybór między kernelem macierz–wektor a macierz–macierz zależy od liczby wierszy na eksperta, a wstępny przebieg `map0` buduje listy tokenów dla każdego eksperta [66].
- **MLX:**
  - Zgłoszenie MLX #4509 dokumentuje niską przepustowość `gather_qmm` przy M = 1 [67].
  - PR #4598 z węższą poprawką odrzucono ze względu na złożoność dyspozycji.
  - PR #4572 poprawił kafelkowanie `gather_qmm` i jest zawarty w MLX 0.32.3+, czyli w naszym punkcie odniesienia.
  - Obsługę wielu zapytań w serwerze mlx-lm (`BatchGenerator`) dodano w PR #626 [68].
- **Wkład tej pracy względem powyższych:**
  - przeniesienie grupowania według eksperta na zunifikowaną pamięć Apple/Metal w MLX, z pomiarem zysku w prawdziwym serwerze;
  - nie znaleźliśmy opublikowanego przeglądu przepustowości MoE dla B = 1–32 na Apple Silicon;
  - kwantyfikacja „gorącego eksperta” na poziomie dokumentu dla Qwen3, gpt-oss i DeepSeek-V2-Lite.

## Bibliografia

Identyfikatory arXiv sprawdzone na arxiv.org (2026-11-07), DOI przez Crossref, adresy URL przez zapytanie HTTP.

[1] A. Vaswani i in., „Attention Is All You Need”, NeurIPS 2017. arXiv:1706.03762.
[2] N. Shazeer i in., „Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”, ICLR 2017. arXiv:1701.06538.
[3] D. Lepikhin i in., „GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding”, ICLR 2021. arXiv:2006.16668.
[4] W. Fedus, B. Zoph, N. Shazeer, „Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity”, JMLR 2022. arXiv:2101.03961.
[5] N. Du i in., „GLaM: Efficient Scaling of Language Models with Mixture-of-Experts”, ICML 2022. arXiv:2112.06905.
[6] B. Zoph i in., „ST-MoE: Designing Stable and Transferable Sparse Expert Models”, 2022. arXiv:2202.08906.
[7] Y. Zhou i in., „Mixture-of-Experts with Expert Choice Routing”, NeurIPS 2022. arXiv:2202.09368.
[8] A. Q. Jiang i in., „Mixtral of Experts”, 2024. arXiv:2401.04088.
[9] D. Dai i in., „DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models”, 2024. arXiv:2401.06066.
[10] DeepSeek-AI, „DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model”, 2024. arXiv:2405.04434.
[11] DeepSeek-AI, „DeepSeek-V3 Technical Report”, 2024. arXiv:2412.19437.
[12] A. Yang i in., „Qwen3 Technical Report”, 2025. arXiv:2505.09388.
[13] OpenAI, „gpt-oss-120b & gpt-oss-20b Model Card”, 2025. arXiv:2508.10925.
[14] N. Shazeer, „GLU Variants Improve Transformer”, 2020. arXiv:2002.05202.
[15] S. Williams, A. Waterman, D. Patterson, „Roofline: An Insightful Visual Performance Model for Multicore Architectures”, Communications of the ACM 52(4), 2009. DOI:10.1145/1498765.1498785.
[16] T. Gale i in., „MegaBlocks: Efficient Sparse Training with Mixture-of-Experts”, MLSys 2023. arXiv:2211.15841.
[17] C. Hwang i in., „Tutel: Adaptive Mixture-of-Experts at Scale”, MLSys 2023. arXiv:2206.03382.
[18] S. Rajbhandari i in., „DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale”, ICML 2022. arXiv:2201.05596.
[19] W. Kwon i in., „Efficient Memory Management for Large Language Model Serving with PagedAttention”, SOSP 2023. arXiv:2309.06180. Kod `fused_moe`: https://github.com/vllm-project/vllm
[20] E. Frantar i in., „MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models”, 2024. arXiv:2408.11743. Kod: https://github.com/IST-DASLab/marlin
[21] DeepSeek-AI, „DeepGEMM”, https://github.com/deepseek-ai/DeepGEMM
[22] NVIDIA, „MoE as Dense GEMM” (TensorRT-LLM tech blog), https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.html
[23] S. Tan i in., „Scattered Mixture-of-Experts Implementation”, 2024. arXiv:2403.08245.
[24] W. Guo i in., „SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations”, 2025. arXiv:2512.14080.
[25] Z. Ye i in., „FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving”, 2025. arXiv:2501.01005. Kod: https://github.com/flashinfer-ai/flashinfer
[26] O. J. Aimuyo i in., „FlashMoE: Fast Distributed MoE in a Single Kernel”, 2025. arXiv:2506.04667.
[27] Y. Gong i in., „MonoMoE: An Efficient Fused Mega-kernel for Quantized MoE Decoding”, 2026. arXiv:2609.04244.
[28] S. Mitra, „Cross-Platform Fused MoE Dispatch in Triton: Portable Expert Routing Without CUDA”, 2026. arXiv:2605.23911.
[29] K. Kamahori i in., „Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models”, 2024. arXiv:2402.07033.
[30] S. Cao i in., „MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs”, 2024. arXiv:2411.11217.
[31] L. Xue i in., „MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache”, 2024. arXiv:2401.14361.
[32] R. Hwang i in., „Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference”, ISCA 2024. arXiv:2308.12066.
[33] Apple, „MLX: An array framework for Apple silicon”, https://github.com/ml-explore/mlx (zgłoszenie #4509 i PR #4598 w tym repozytorium).
[34] Apple, „Custom Metal Kernels” (dokumentacja MLX), https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
[35] G. Gerganov i in., „llama.cpp”, https://github.com/ggml-org/llama.cpp
[36] F. Waschkowski i in., „BaseRT: Advancing Best-in-Class LLM Inference with Apple M5 Neural Accelerators”, 2026. arXiv:2607.19438.
[37] A. Benazir, F. X. Lin, „Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs” (NPUMoE), 2026. arXiv:2604.18788.
[38] A. Novikov i in., „AlphaEvolve: A coding agent for scientific and algorithmic discovery”, 2025. arXiv:2506.13131.
[39] A. Ouyang i in., „KernelBench: Can LLMs Write Efficient GPU Kernels?”, 2025. arXiv:2502.10517.
[40] A. Karpathy, „autoresearch”, https://github.com/karpathy/autoresearch
[41] J. Jaber, O. Jaber, „AutoKernel: Autonomous GPU Kernel Optimization via Iterative Agent-Driven Search”, 2026. arXiv:2603.21331.
[42] S. Cao i in., „K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model”, 2026. arXiv:2602.19128.
[43] X. Li i in., „CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning”, 2025. arXiv:2507.14111.
[44] B. D. Rouhani i in., „Microscaling Data Formats for Deep Learning”, 2023. arXiv:2310.10537.
[45] P. Micikevicius i in., „FP8 Formats for Deep Learning”, 2022. arXiv:2209.05433.
[46] E. Frantar i in., „GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers”, ICLR 2023. arXiv:2210.17323.
[47] J. Lin i in., „AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration”, MLSys 2024. arXiv:2306.00978.
[48] P. Tillet, H. T. Kung, D. Cox, „Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations”, MAPL 2019. DOI:10.1145/3315508.3329973.
[49] T. Dao i in., „FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”, NeurIPS 2022. arXiv:2205.14135.
[50] Y. Leviathan, M. Kalman, Y. Matias, „Fast Inference from Transformers via Speculative Decoding”, ICML 2023. arXiv:2211.17192.
[51] C. Chen i in., „Accelerating Large Language Model Decoding with Speculative Sampling”, 2023. arXiv:2302.01318.
[52] Apple, „Apple introduces M4 Pro and M4 Max” (komunikat prasowy, 10.2024), https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/
[53] Apple, „Metal Feature Set Tables”, https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
[54] Apple, „Metal Shading Language Specification”, https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
[55] NVIDIA, „H100 Tensor Core GPU”, https://www.nvidia.com/en-us/data-center/h100/
[56] JAX Authors, „Pallas: a JAX kernel language”, https://docs.jax.dev/en/latest/pallas/index.html
[57] J. Austin i in., „How to Scale Your Model”, https://jax-ml.github.io/scaling-book/
[58] OpenXLA, „Tokamax”, https://github.com/openxla/tokamax (⚠ projekt we wczesnej fazie; zakres funkcji niezweryfikowany w dokumentacji)
[59] Google, „MaxText” (zawiera Megablox), https://github.com/AI-Hypercomputer/maxtext
[60] P. Vellaisamy i in., „TaxBreak: Unmasking the Hidden Costs of LLM Inference Through Overhead Decomposition”, 2026. arXiv:2603.12465.
[61] JAX Authors, „JAX: composable transformations of Python+NumPy programs”, https://github.com/jax-ml/jax
[62] Apple, „Explore GPU advancements in M3 and A17 Pro” (Tech Talk), https://developer.apple.com/videos/play/tech-talks/111375/
[63] J. Liang i in., „Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models”, 2025. arXiv:2505.16056.
[64] Z. Qiu i in., „Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models”, ACL 2025. arXiv:2501.11873. (⚠ treść znana z raportu agenta; do weryfikacji)
[65] Z. Su i in., „Unveiling Super Experts in Mixture-of-Experts Large Language Models”, ICLR 2026. arXiv:2507.23279.
[66] ggml-org/llama.cpp, PR #12612 (mul_mv_id vs mul_mm_id wg liczby wierszy na eksperta), https://github.com/ggml-org/llama.cpp/pull/12612 (⚠ do weryfikacji)
[67] ml-explore/mlx, zgłoszenie #4509, PR #4598, PR #4572, https://github.com/ml-explore/mlx
[68] ml-explore/mlx-lm, PR #626 (batching w serwerze), https://github.com/ml-explore/mlx-lm/pull/626 (⚠ do weryfikacji)
