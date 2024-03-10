import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Float, String, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

engine = create_engine(
    os.getenv('DB_URL'),
    echo=True, 
    pool_size=5, 
    max_overflow=-1,
    pool_recycle=3600, 
    pool_pre_ping=True, 
    connect_args={
        "connect_timeout": 300,
        "keepalives": 1, 
        "keepalives_idle": 300, 
        "keepalives_interval": 100, 
        "keepalives_count": 5,
    },
)

Session = sessionmaker(bind=engine)
connection = engine.connect()

Base = declarative_base()

class Fraud(Base):
    __tablename__ = "fraud"
    id = Column(Integer, primary_key=True)
    trans_date_trans_time = Column(String)
    cc_num = Column(BigInteger)
    merchant = Column(String)
    category = Column(String)
    amt = Column(Float)
    first = Column(String)
    last = Column(String)
    gender = Column(String)
    street = Column(String)
    city = Column(String)
    state = Column(String)
    zip = Column(Integer)
    lat = Column(Float)
    long = Column(Float)
    city_pop = Column(Integer)
    job = Column(String)
    dob = Column(String)
    trans_num = Column(String)
    unix_time = Column(Integer)
    merch_lat = Column(Float)
    merch_long = Column(Float)
    is_fraud = Column(Integer)

Base.metadata.create_all(bind=engine)
connection.close()
