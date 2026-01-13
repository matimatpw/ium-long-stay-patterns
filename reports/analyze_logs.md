# Raport z Testów A/B:

## 1. Metryki Ogólne
Analiza została przeprowadzona na podstawie logów systemowych zawierających **1368 wpisów**. System wykazał się pełną stabilnością operacyjną.

* **Data analizy:** 13 stycznia 2026 r.
* **Całkowita liczba żądań:** 1368
* **Wskaźnik sukcesu (Success Rate):** 100% (brak błędów w obu modelach)

---

## 2. Porównanie Modeli

W teście zestawiono model bazowy (**NAIVE**) z modelem uczenia maszynowego (**BINARY**).

| Metryka | Model NAIVE (Kontrola) | Model BINARY (Test) | Zmiana |
| :--- | :---: | :---: | :---: |
| **Liczba próbek** | 666 | 702 | +5.4% |
| **Predykcje Longstay** | 0 | 202 | +202 |
| **Wskaźnik Longstay** | 0.00% | 28.77% | +28.77 pp |
| **Conversion Rate** | 0.00% | **28.77%** | **+28.77%** |
| **Średnia predykcja** | 0.0000 | 0.2877 | +0.2877 |

---

## 3. Analiza Statystyczna
Wyniki porównania modeli wykazują jednoznaczną przewagę modelu BINARY pod kątem zdolności predykcyjnych.

* **Z-Score:** 14.9947
* **P-Value:** 0.0000
* **Istotność statystyczna (α=0.05):** **TAK**
* **Test Chi-kwadrat:** 222.56 (potwierdza silną zależność między modelem a wynikami)

### Wnioski z testów:
1.  **Model NAIVE** jest modelem pasywnym – nie identyfikuje żadnych rezerwacji typu "Longstay", co czyni go nieużytecznym w celach optymalizacji biznesowej.
2.  **Model BINARY** aktywnie klasyfikuje rezerwacje, przypisując status "Longstay" do ok. 28.8% przypadków.
3.  **Wysoka istotność statystyczna** (P-Value < 0.05) pozwala odrzucić hipotezę, że różnica między modelami wynika z przypadku.

---

## 4. Rekomendacja Biznesowa
Na podstawie zebranych danych zaleca się **pełne wdrożenie modelu BINARY**. Model ten dostarcza realną wartość analityczną i pozwala na segmentację klientów pod kątem długości pobytu, co było niemożliwe przy zastosowaniu podejścia Naiwnego.

---
*Analiza dotyczy danych w pliku pliku: `logs_ab.txt`*