% Fakty
choroba(grypa) :- objaw(goraczka), objaw(kaszel), objaw(bol_glowy).
choroba(przeziebienie) :- objaw(kichanie), objaw(kaszel), objaw(zatkany_nos).
choroba(angina) :- objaw(bol_gardla), objaw(goraczka), objaw(trudnosci_z_polknieciem).
choroba(migrena) :- objaw(bol_glowy), objaw(zawroty_glowy).
choroba(alergia) :- objaw(kichanie), objaw(swedzenie_oczu), objaw(zaczerwienienie_skory).
choroba(zapalenie_oskrzeli) :- objaw(kaszel), objaw(duszności).

% Reguły interakcji
pytaj_o_objawy :-
    write('Czy uwazasz, ze jestes chory? (tak/nie)'), nl,
    read(Odpowiedz),
    Odpowiedz == tak.

objaw(Objaw) :-
    write('Czy masz objaw: '), write(Objaw), write('? (tak/nie)'), nl,
    read(Odpowiedz),
    Odpowiedz == tak.

% Uruchomienie diagnozy
diagnoza :-
    pytaj_o_objawy,
    choroba(Choroba),
    write('Mozliwa diagnoza to: '), write(Choroba), nl.

diagnoza :-
    write('Nie moge ustalic diagnozy na podstawie podanych objawow.'), nl.