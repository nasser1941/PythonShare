from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from SQL.mysqlac import Base


def connect2DB(connectionString):
    print "Connection to db: ", connectionString
    engine = create_engine(connectionString)
    Base.metadata.create_all(engine)
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    connect2DB.session = session
    return session