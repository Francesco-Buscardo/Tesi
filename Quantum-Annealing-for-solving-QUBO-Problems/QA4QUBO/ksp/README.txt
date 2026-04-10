Formato ksp_n file:
- n_items: numero items
- capacity: C
- items: weight profit
----------------------[ 1 ]----------------------
./ksp_1.txt:
10 
101 

42 81 
42 42 
68 56 
68 25 
77 14 
57 63 
17 75 
19 41 
94 19 
34 12

-------------------[ Results ]---------------------
In 1000 runs on ./ksp_1.txt 
--[ Avg Weight GAP ]--[ 90.6 ]
--[ Avg Profit GAP ]--[ 10.7 ]

-----------------[ Conclusioni ]------------------
In 1000 esecuzioni su ksp_1.txt, QALS ha ottenuto 
un profit gap medio del 10.7%, ma un weight gap 
medio di 90.6 unità. Poiché la capacità dello zaino 
è 101, ciò significa che in media le soluzioni 
superano il vincolo di quasi un’intera capacità 
aggiuntiva. Questo indica che l’attuale formulazione 
QUBO a vincoli soft non riesce a imporre efficacemente 
l’ammissibilità, pur trovando talvolta soluzioni 
con profitto relativamente vicino all’ottimo.

-------------[ Soluzione Possibile ]---------------
Dopo che si è preso la soluzione si guarda tra 
gli oggeti non selezionati e si aggiungono allo 
zaino partendo da quello da quello con profitto 
maggiore.
-------------------[ End 1 ]-----------------------