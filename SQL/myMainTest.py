from SQL.mysqlac import Utente
from connection2DB import connect2DB

session = connect2DB("sqlite:///testAlchemy.db")
utente = Utente(name = u"pippo", surname = u"Pluto")

listaUtenti = Utente.getAllUsers(session=session)
for utente in listaUtenti:
    print utente.getMap()

#def addUser():


print utente.__dict__
try:
    session.add(utente)
    session.commit()
    print utente.__dict__
    print "New Entry added"

except Exception, e:
    session.rollback()
    print e.message