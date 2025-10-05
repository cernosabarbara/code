*create and recode variables* 

*tabulation of demands, question Q15D in the HRS* 
tab PLB015D 
*tabulation without labels* 
tab PLB015D, nol 
*creating new variable* 
gen deman=PLB015D 
*recode* 
recode deman (1=4) (2=3) (3=2) (4=1) (else=.) 
*compare* 
tab1 PLB015D deman 
 
*tabulation of critique, question Q15E in the HRS* 
tab PLB015E 
*tabulation without labels* 
tab PLB015E, nol 
*creating new variable* 
gen crit=PLB015E 
*recode* 
recode crit (1=4) (2=3) (3=2) (4=1) (else=.) 
*compare* 
tab1 PLB015E crit 

*tabulation of down, question Q15F in the HRS* 
tab PLB015F 
*tabulation without labels* 
tab PLB015F, nol 
*creating new variable* 
gen down=PLB015F 
*recode* 
recode down (1=4) (2=3) (3=2) (4=1) (else=.) 
*compare* 
tab1 PLB015F down 

*tabulation of nerves, question Q15G in the HRS* 
tab PLB015G 
*tabulation without labels* 
tab PLB015G, nol 
*creating new variable* 
gen nerve=PLB015G 
*recode* 
recode nerve (1=4) (2=3) (3=2) (4=1) (else=.) 
*compare* 
tab1 PLB015G nerve 

*calculates numbers of questions missing for negints* 
egen missing=rowmiss(deman crit down nerve) 
*labels variable overall* 
label variable missing "Number of missing for negints" 
*tabulation* 
tab missing 

*calculates mean of the negints if missing less than 1 of the items* 
egen negints=rmean(deman crit down nerve) if missing<=1 
*labels variable overall* 
label variable negints "Negative Interactions" 
*tabulation* 
tab negints 
*mean and sd* 
sum negints 

*create and recode variables* 
*tabulation of anyfriends, question Q14 in the HRS* 
tab PLB014 
*tabulation without labels* 
tab PLB014, nol 
*creating new variable* 
gen anyfriends=PLB014 
*recode* 
recode anyfriends (1=1) (5=0) (else=.) 
*labels variable overall* 
label variable anyfriends "Do you have any friends" 
*creates labels* 
label define anyfriendslab 0 "No friends" 1 "Friends" 
*labels categories* 
label values anyfriends anyfriendslab 
*compare* 
tab1 PLB014 anyfriends 

*tabulation of number of negative interactions, including those without friends* 
*tabulation of number of negative interactions* 
tab negints 
*tabulation without labels* 
tab negints, nol 
*creates label for those without friends* 
gen negfriends=negints  
*replaces response with 1, if they have 0 friends* 
replace negfriends=1 if anyfriends==0 
*labels variable overall* 
label variable negfriends "Negative Interactions, 1 if no friends" 
*compare* 
tab1 negints negfriends 
*tabulate negfriends* 
tab negfriends 

*crosstab of negfriends and anyfriends* 
tab negfriends anyfriends, m 

*create and recode shame variable*
*tabulation of shame, question Q26O in the HRS*
tab PLB026O
*tabulation without labels*
tab PLB026O, nol
*creating new variable*
gen shame=PLB026O
*recode*
recode shame (1=5) (2=4) (3=3) (4=2) (5=1) (else=.)
*labels variable overall*
label variable shame "Ashamed"
*compare*
tab1 PLB026O shame

*create and recode education variable*
*tabulation of education, raedyrs in the HRS*
tab raedyrs
*tabulation without labels*
tab raedyrs, nol
*creating new variable*
gen education=raedyrs if raedyrs>=0 & raedyrs<20
*labels variable overall*
label variable education "Education"
*compare*
tab1 raedyrs education

*create sss var* 
*tabulation, question Q36 in the HRS*
tab PLB036
*tabulation without labels*
tab PLB036, nol
*creating new variable*
gen sss=PLB036 if PLB036>0 & PLB036<11
*labels variable overall*
label variable sss "SSS"
*tabulation before and after recoding*
tab1 PLB036 sss

*create age var* 
*tabulation, PAGE in the HRS*
tab PAGE
*tabulation without labels*
tab PAGE, nol
*creating new variable*
gen age=PAGE if PAGE>49 & PAGE<99
*labels variable overall*
label variable age "Age"
*tabulation before and after recoding*
tab1 PAGE age


*creates flag* 
gen flag=0 
replace flag=1 if shame!=. & education!=. & negfriends!=. & sss!=. & age!=. 
*tabulate flag* 
tab flag 
*regression for the association between shame and negfriends* 
*controls for sss age and education* 
regress shame negfriends age education if flag==1, beta 


*regression for the association between shame and negfriends* 
*controls for sss age and education* 
regress shame negfriends age education sss if flag==1, beta 
*obtains vif* 
vif 


 