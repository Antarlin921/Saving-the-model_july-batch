import joblib
# load the model
loaded_model=joblib.load('dib_79.pkl')
# names=['preg','plas','pres','skin','test','mass','pedi','age','class']
# preg=
# plas=
# pres=
# skin=
# test=
# mass=
# predi=
# age=
pred=loaded_model.predict([[10,20,30,40,50,10,20,10]])
print(pred)
if pred[0]==1:
    print('The person is Diabetic')
else:
    print('The person is not Diabetic')
