class Utente():
    name = ""
    surname = ""
    address = ""

    def __init__(self): # constructor - optional in Python
        self.name = "Naser"
        self.surname = "Derakhshan"

    def getMap(self):
        return {"name: ": self.name, "sur: ": self.surname, "address: " :self.address}

    @classmethod  #to define static class
    def addUser(cls, name, surname): # cls is like self
        utente = Utente()
        utente.name = name
        utente.surname = surname
        print cls.printUtente()
        return utente

    @classmethod
    def printUtente(cls):
        print "Ciao"
        return 'returned' # if we do not put return, it will return none

"""
For block comments
Use this
"""

## for doxygen
#@param self
#@return a dictionary of the class

utente = Utente()
print utente.__dict__
print utente.getMap()
utente2 = Utente.addUser(name = "sss", surname='ssdd')
print utente2.__dict__