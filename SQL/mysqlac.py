from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import Time
from sqlalchemy import UnicodeText
from sqlalchemy.ext.declarative import declarative_base
#sqlalchemy.org for documentation

Base = declarative_base()

class Utente(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, )
    name = Column(UnicodeText(20),
                  nullable=False,
                  default=None,
                  unique=False)
    surname= Column(UnicodeText(20))
    time=Column(Time)


    @classmethod
    def getAllUsers(clscls, session):
        print session.query(Utente)
        q=session.query(Utente).all()
        return q

    def getMap(self):
        return {"name: ": self.name, "surname ": self.surname, "time ": self.time}

    @classmethod
    def searchUser(cls, session, name =None, surname=None):
        q=session.query(Utente)
        q2 = session.query(Utente).filter(Utente.name==name).filter(Utente.surname==surname)
        if name is not None:
            q=q.filter(Utente.name==name)
        if surname is not None:
            q=q.filter(Utente.surname==surname)
        return q.all()


   # session.query(Utente).filter(Utente)
