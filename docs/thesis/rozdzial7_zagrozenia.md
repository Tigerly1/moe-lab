# Rozdział 7. Zagrożenia dla trafności wyników

> Szkic. Każdy punkt opisuje błąd, który rzeczywiście popełniliśmy lub wykryliśmy w trakcie badań, oraz sposób jego usunięcia.

## 7.1. Nieuczciwy punkt odniesienia (wykryty przez audyt)

**Błąd.** Pierwsza baza była ręczną kopią ścieżki mlx-lm. Liczyła SwiGLU trzema osobnymi operacjami zamiast jednej skompilowanej i dodawała zbędne rzutowanie typu. Zawyżało to przyspieszenie: raportowaliśmy 1,25× zamiast 1,17×.

**Środek zaradczy.** Baza wywołuje teraz bezpośrednio moduł `mlx_lm.models.switch_layers.SwitchGLU`. Wynik end-to-end był od początku liczony na prawdziwej ścieżce mlx-lm i potwierdza poprawioną liczbę.

## 7.2. Artefakty pomiaru czasu

1. **Zanikające aktywacje.** Pomiar łańcuchowy bez strumienia rezydualnego zerował aktywacje już po około 3 wywołaniach. Dodaliśmy rezyduum (x ← x + y).
2. **Stały koszt `eval`.** Na M4 Pro wynosi około 130 µs po stronie hosta i 130–340 µs łącznie. Rozłożony na łańcuch, zaniżał przyspieszenia. Czas jednego wywołania wyznaczamy dlatego jako nachylenie względem długości łańcucha.
3. **Szum bazy.** Czasy MLX przy T = 4–8 wahają się o około ±15% między uruchomieniami. Reguła akceptacji w pętli autoresearch, oparta na przedziale ufności bazy, odrzuciła przez to trzy lepsze kernele. Kryterium zmieniliśmy na przedział ufności kandydata.
4. **Dekodowanie spekulatywne.** Szybkość tok/s przy swobodnej generacji nie porównuje samych kerneli. Gdy oba warianty wygenerują różny tekst (szum bf16), zmienia się odsetek akceptacji draftu: 0,63 wobec 0,54. Wprowadziliśmy benchmark weryfikacji, w którym oba warianty przetwarzają ten sam tekst kawałkami.

## 7.3. Uogólnienie na inne układy

**Taktowanie i temperatura.** Wykluczyliśmy wpływ boostu zegara:
- 10 minut ciągłego obciążenia: stosunek czasów 1,169 → 1,167, bez dryfu czasów bezwzględnych;
- zasilanie bateryjne i sieciowe dają ten sam zysk: +7,1% i +7,0%.

**Inne generacje układów Apple** nie zostały przetestowane:
- **Próg K % 512:** ta część efektu wynika z kodu MLX niezależnego od układu, więc spodziewamy się jej także na M1–M3.
- **Fuzja i obsadzenie:** ta część może zależeć od Dynamic Caching w M3 i M4.
- **M5:** MLX używa tam innych ścieżek (NAX, akceleratory neuronowe GPU).

## 7.4. Reprezentatywność routingu

**Zakres śladów.** Ślady routingu zebraliśmy dla jednego modelu (Qwen3-30B-A3B) z około 5,4 tys. tokenów w 3 domenach: angielskiej, kodu i polskiej.

**Ograniczenia, których nie sprawdziliśmy:**
- inne modele, np. OLMoE, DeepSeek i gpt-oss, mogą mieć inną lokalność;
- długie konteksty i dialogi mogą zmieniać rozkład wyboru ekspertów.

**Najsilniejsze wnioski** (zysk przy T > 1) dotyczą kolejnych tokenów jednej sekwencji, co odpowiada weryfikacji w dekodowaniu spekulatywnym. Dla tokenów z różnych zapytań (batching) lokalność jest mniejsza (tabela w §4.4).

## 7.5. Autonomiczne agenty

Część kerneli powstała w pętli autoresearch (agent LLM). Zabezpieczenia przed oszukiwaniem metryki:
- edycja tylko jednego pliku;
- ewaluator tylko do odczytu, z bramką poprawności dla trzech rodzajów routingu, w tym one-hot;
- losowe wejścia przy każdym uruchomieniu;
- świeży proces na każdy pomiar.

Każdy wynik agenta zweryfikowaliśmy niezależnie, z większą liczbą powtórzeń. Raz agent zaniżył wynik (nie zmierzył T = 32). Raz jego reguła akceptacji odrzuciła poprawny kernel.

## 7.6. Wagi syntetyczne a rzeczywiste checkpointy

**Błąd.** Kernel MXFP4 przechodził wszystkie testy na losowych wagach, a mimo to był błędny dla prawdziwego checkpointu gpt-oss. Ścieżka MMA zapisywała skalę E8M0, przemnożoną przez 2^14, w jednej liczbie typu half. Przy bajtach skali ≥ 129 wartość ta przepełniała zakres half i dawała inf. W rzeczywistym gpt-oss dotyczy to około 11% grup w projekcji down. Wagi losowe o rozkładzie normalnym nigdy nie dają tak dużych skal.

**Wykrycie i poprawka.** Błąd ujawnił dopiero test na pełnym modelu. Poprawka rozbija skalę na dwa czynniki half, z których każdy jest dokładny. Test regresyjny `tests/test_mxfp4_large_scales.py` wymusza duże bajty skali i obejmuje wszystkie trzy reżimy (T = 1, małe T, T ≥ 8).

**Wpływ na wyniki.** Poprawność kosztuje 8–11% przyspieszenia przy T ≥ 16.

**Wniosek metodyczny.** Bramka poprawności musi obejmować rozkłady wag z prawdziwych modeli, nie tylko wagi losowe.

## 7.7. Zależność statystyczna pomiarów i „held-out”

**Zbyt wąskie przedziały ufności.** Drugi niezależny audyt wykazał, że przy routingu ze śladów przedziały ufności były zbyt wąskie. Bootstrap traktował rundy jako niezależne, choć rundy w ramach jednego uruchomienia współdzielą te same okna routingu. Dla T = 8 przedział ±1% po poprawnym bootstrapie po uruchomieniach rozszerza się do [1,63, 1,83].

**Poprawka metody:**
- bootstrap hierarchiczny (najpierw uruchomienia, potem rundy);
- 8 okien routingu na każde uruchomienie;
- ziarna losowe, których agenty strojące nigdy nie widziały.

Estymaty punktowe się utrzymały.

**Kształty „held-out”.** Okazało się, że w pełni czysty jest tylko Qwen3-235B:
- Mixtral i DeepSeek-V2-Lite pojawiały się wcześniej w benchmarkach strojenia;
- gpt-oss-120b ma ten sam kształt eksperta co gpt-oss-20b.

**Cache w łańcuchu pomiarowym.** Łańcuch pomiarowy obracał tylko 3 wejścia, przez co zestawy ekspertów powtarzały się co trzecie wywołanie i część odczytów trafiała do cache. Przy niepowtarzającym się routingu odsetek sufitu dla Qwen3 spada z 98,5% do 93,7%. Przyspieszenie zmienia się nieznacznie: z 1,24× na 1,22×.

## 7.8. Długość kontekstu

Zysk z dekodowania mierzyliśmy głównie przy krótkich promptach. Przy dłuższym kontekście rośnie udział atencji, czyli odczytów pamięci podręcznej KV, więc zysk maleje zgodnie z prawem Amdahla. Dla Qwen3-30B-A3B przy 8 tys. tokenów kontekstu wynosi on +4,3%, wobec +9,0% przy krótkim promptcie. Wnioski o warstwie MoE pozostają bez zmian, ale korzyść w pełnym systemie zależy od profilu obciążenia.

## 7.9. Zakłócenia od innych procesów na GPU (runda 7)

- **Problem:** inne aplikacje (przeglądarka, kompozytor ekranu) obniżają bezwzględne liczby. Pomiar T = 1 dał 80% granicy przepustowości zamiast 93%, a stosunek A/B prawie się nie zmienił (1,196× wobec 1,222×).
- **Strażnik:** każdy pomiar jest otoczony sondą przepustowości odczytu, porównywaną z 90. percentylem ostatnich sond dla danego układu i źródła zasilania. Przebiegi poniżej 92% są oznaczane jako zakłócone.
- **Ograniczenia strażnika:**
  - mierzy tylko przepustowość pamięci, więc nie wykrywa każdego zakłócenia (§7.10);
  - przy długotrwałym obciążeniu okno historii po kilku przebiegach uznaje zakłócenie za normę.

## 7.10. Wyniki nieodtwarzalne i fałszywe alarmy

- **Fałszywe alarmy przesiewu.** Szybki przesiew w jednym procesie zgłosił kilka „strat” (0,51×, 0,55×, 0,78×), które w pełnej procedurze okazały się wygranymi 1,05–1,31×. Zasada: przesiew tylko nominuje przypadki, a liczą się wyłącznie ponowne pomiary.
- **Nieodtworzony spadek przy długim kontekście.** Pojedynczy pomiar w ciągu dnia sugerował spadek zysku MoE do 1,04× przy 16k. Dwa czyste powtórzenia dały 1,13–1,14×. Strażnik tamtego przebiegu nie oznaczył.
- **Wartości bezwzględne zależą od stanu maszyny.** Wartości bezwzględne tok/s zmieniały się w ciągu dnia o około 9% dla obu wariantów naraz, przy stałym stosunku. Porównujemy wyłącznie stosunki z tej samej sesji.

## 7.11. Wycofane twierdzenie o routerze

- **Pierwotne twierdzenie:** w 9% przypadków router mlx-lm wybiera ekspertów arbitralnie, bo zaokrągla prawdopodobieństwa do bf16 przed top-k, a nasz kernel wybiera „lepiej”.
- **Weryfikacja:** sprawdziliśmy stock z jego własnym kodem wyboru. Zawsze zwraca poprawny top-k według logitów (0 błędów na 24 576 przypadków). Różnice to dokładne remisy samych logitów bf16, których żadna precyzja softmaxu nie usuwa.
- **Wniosek:** twierdzenie wycofano, a planowane zgłoszenie do mlx-lm porzucono. Pierwsza wersja weryfikacji również zawierała błąd (forma wyboru z DeepSeek zastosowana do Qwen3).

## 7.12. Ograniczenia porównania z llama.cpp i z serwerem

- llama.cpp i serwer mlx-lm porównujemy wyłącznie w ustawieniach domyślnych, na jednym modelu.
- Formaty Q4_0 i MLX 4-bit g64 mają tę samą liczbę bitów, ale różną strukturę skal.
- W trakcie pomiarów serwera wykryliśmy dwa własne błędy narzędzia: puste prompty oraz nieczytany potok logów blokujący serwer. Wyniki zmierzyliśmy ponownie po poprawkach. Przy okazji potwierdziliśmy błąd mlx-lm 0.32.2: jedno zapytanie z pustym promptem trwale wyłącza wątek generowania serwera. Ta klasa błędów jest już zgłoszona (#1513, #1909).

