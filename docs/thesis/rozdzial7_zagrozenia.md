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
