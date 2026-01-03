1. Jak obsłużyć opcje bookingu a następnie jego anulowania (czy zakladać ze anulowanie nastąpiło przed rozpoczęciem wynajmu)




# TODO

<!-- 1. listing_statistics -->
2. amenities standaryzowac
<!-- 3. polaczyc w 1 data set interesujace kolumny -->
4. host_response_time (zestandaryzowac jakos jesli sie da chyba tak)

5. feature importance (cechy z listings) zrobic

6. zrobic duzy dataset OHE do modelu xgboost/BinaryClassifier i wytrenowac (punkt 2 i 4)
7. microservice: base - naive / big_model - xgboost/BinaryClassifier



# QUESTIONS
1. czy binarny i naiwny wystarcza czy moze jeszcze jakiego xgboosta wytrenowac?
2. czy nie ma wiecej danych bo listingow mamy 1400 (po podziale na tre/val/test trenujemy na ok 900 probkach z wiec przy balansie 40% mamy jakos 300 klas pozytywnych)
3. testy AB (co to)?
4. czy robic te feature importance czy macierz korelacji wystarczy?
5. co jeszcze brakuje do oddania?