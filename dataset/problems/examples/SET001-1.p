%--------------------------------------------------------------------------
% File     : SET001-1 : TPTP v9.2.1. Released v1.0.0.
% Domain   : Set Theory
% Problem  : Set members are superset members
% Version  : [LS74] axioms.
% English  : A member of a set is also a member of that set's supersets.

% Refs     : [LS74]  Lawrence & Starkey (1974), Experimental Tests of Resol
%          : [WM76]  Wilson & Minker (1976), Resolution, Refinements, and S
% Source   : [SPRFN]
% Names    : ls100 [LS74]
%          : ls100 [WM76]

% Status   : Unsatisfiable
% Rating   : 0.00 v2.0.0
% Syntax   : Number of clauses     :    9 (   3 unt;   1 nHn;   8 RR)
%            Number of literals    :   17 (   0 equ;   8 neg)
%            Maximal clause size   :    3 (   1 avg)
%            Maximal term depth    :    2 (   1 avg)
%            Number of predicates  :    3 (   3 usr;   0 prp; 2-2 aty)
%            Number of functors    :    4 (   4 usr;   3 con; 0-2 aty)
%            Number of variables   :   13 (   0 sgn)
% SPC      : CNF_UNS_RFO_NEQ_NHN

% Comments :
%--------------------------------------------------------------------------
%----Include the member and subset axioms
include('Axioms/SET001-0.ax').
%--------------------------------------------------------------------------
cnf(b_equals_bb,hypothesis,
    equal_sets(b,bb) ).

cnf(element_of_b,hypothesis,
    member(element_of_b,b) ).

cnf(prove_element_of_bb,negated_conjecture,
    ~ member(element_of_b,bb) ).

%--------------------------------------------------------------------------
