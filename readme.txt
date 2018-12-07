Goal:
This code is part of a UZH course project named: 
"Introduction to systematic risk premia strategies traded at hedge funds"
Our goal for the project was to collect data from 2 exchanges and 1 web site 
place them into a database and retrieve them to create an arbitrage index graph.
Our ultimate goal is to make this program to trade, but unfortunately we 
were not able to establish an account in Bithumb (Korea) which was a necessary
assumption to replicate the paper that we presented. 
In the future we will further develop this program to make it fully functional.


Implementation Details:
We do the json queries in two different ways, because we tried to import a pycurl
but we encountered some problems since we were working on windowns. To solve
this problem we tried to find compiled the the file but we couldn't. Therefore,
for Bithumb we used traditional json queries while for Coinbase we used 
"xcoin_api_client" functions. To find the exchange rate between USD and KRW
we used the information from the website "fixer", for which we used again
traditionally json queries. For the version we uploaded we let the database
empty. The user should create a table and run the program to save the new
data. 

