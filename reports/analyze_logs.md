# Analiza logów po testach A/B

## Porówanie modeli Baseline(Naive) vs Binary

1. Model binarny przewiduje 25.21% więcej długich pobytów niż model naiwny.

1. Model naiwny przewiduje zawsze klase wiekszościową (shortstay) więc list=0, a zatem model binarny jest nieskończenie lepszy od modelu naiwnego.

3. Oba modele przeprocesowały każde żądanie, stąd succes_rate = 100%

4. Z-score = 6.62 i p-value ≈ 0 oznaczają, że różnica między modelami jest statystycznie istotna z pewnością >99.99% (różnica między modelami nie jest zaskoczeniem.)

5. *significant* mówi nam, że róznica między modelami jest faktycznie istotna.
(Modele działają inaczej, niezależnie od danych)

6. Test Chi-kwadrat (χ² = 41.33, p < 0.0001) potwierdza wyniki testu - różnica w wzorcach predykcji między modelami jest statystycznie istotna i nie wynika z przypadku.

