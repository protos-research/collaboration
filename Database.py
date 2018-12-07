import os
import sqlite3

# tutorial points for sqlite3
# https://www.tutorialspoint.com/sqlite/sqlite_select_query.htm

# Define Custom import vars

path = 'C:\\Users\\thoma\\BithumbTrader\\db\\ExchangesDB.db'

conn = sqlite3.connect(path, check_same_thread=False)


class Database():
    # Write into the database methods
    @staticmethod
    def WriteHourlyData(data):
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO Exchanges (DATE, USDKRW, BUY_BTCUSD_KRW, SELL_BTCUSD_KRW, BUY_BTCUSD, SELL_BTCUSD, abs_diff, "
            "abs_diff_percentage)) VALUES (%s,%f,%f,%f,%f,%f,%f,%f)" % (data[0], data[1], data[2], data[3],
                                                                        data[4], data[5], data[6], data[7]))
        conn.commit()

    @staticmethod
    def WriteOrders(data):
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)''', data)
        conn.commit()

    # Read from database methods
    @staticmethod
    def ReadExchanges(orderid):
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM Exchanges')
        return cur.fetchone()

    @staticmethod
    def ReadOrders(orderid):
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM orders)
        return cur.fetchone()
