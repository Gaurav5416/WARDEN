from index import Fraud, Session

with Session.begin() as db:
    result = db.query(Fraud).all()
    for row in result :
        print(row.is_fraud)