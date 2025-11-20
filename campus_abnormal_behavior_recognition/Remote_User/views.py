from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,student_behavior_prediction,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Campus_Behavior_Type(request):
    if request.method == "POST":

        if request.method == "POST":


            Sid= request.POST.get('Sid')
            Certification_Course= request.POST.get('Certification_Course')
            Gender= request.POST.get('Gender')
            Department= request.POST.get('Department')
            Height_CM= request.POST.get('Height_CM')
            Weight_KG= request.POST.get('Weight_KG')
            Tenth_Mark= request.POST.get('Tenth_Mark')
            Twelth_Mark= request.POST.get('Twelth_Mark')
            college_mark= request.POST.get('college_mark')
            hobbies= request.POST.get('hobbies')
            daily_studing_time= request.POST.get('daily_studing_time')
            prefer_to_study_in= request.POST.get('prefer_to_study_in')
            degree_willingness= request.POST.get('degree_willingness')
            social_medai_video= request.POST.get('social_medai_video')
            Travelling_Time= request.POST.get('Travelling_Time')
            Stress_Level= request.POST.get('Stress_Level')
            Financial_Status= request.POST.get('Financial_Status')
            part_time_job= request.POST.get('part_time_job')
            Campus_Behavior= request.POST.get('Campus_Behavior')


        df = pd.read_csv('Student_Behaviour.csv')

        def apply_response(Label):
            if (Label == 0):
                return 0  # Proper Behavior
            elif (Label == 1):
                return 1  # Improper Behavior

        df['results'] = df['Label'].apply(apply_response)

        cv = CountVectorizer()
        X = df['Sid']
        y = df['results']

        print("Sid")
        print(X)
        print("Results")
        print(y)

        X = cv.fit_transform(X)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print(X_test)

        print("Convolutional Neural Networks (CNNs)")
        from sklearn.neural_network import MLPClassifier
        mlpc = MLPClassifier().fit(X_train, y_train)
        y_pred = mlpc.predict(X_test)
        testscore_mlpc = accuracy_score(y_test, y_pred)
        accuracy_score(y_test, y_pred)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('MLPClassifier', mlpc))


        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print(naivebayes)
        print(confusion_matrix(y_test, predict_nb))
        print(classification_report(y_test, predict_nb))
        models.append(('naive_bayes', NB))


        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))


        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))


        print("Gradient Boosting Classifier")

        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
            X_train,
            y_train)
        clfpredict = clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, clfpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, clfpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, clfpredict))
        models.append(('GradientBoostingClassifier', clf))


        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        Sid1 = [Sid]
        vector1 = cv.transform(Sid1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if (prediction == 0):
            val = 'Proper Behavior'
        elif (prediction == 1):
            val = 'Improper Behavior'


        print(val)
        print(pred1)

        student_behavior_prediction.objects.create(
        Sid=Sid,
        Certification_Course=Certification_Course,
        Gender=Gender,
        Department=Department,
        Height_CM=Height_CM,
        Weight_KG=Weight_KG,
        Tenth_Mark=Tenth_Mark,
        Twelth_Mark=Twelth_Mark,
        college_mark=college_mark,
        hobbies=hobbies,
        daily_studing_time=daily_studing_time,
        prefer_to_study_in=prefer_to_study_in,
        degree_willingness=degree_willingness,
        social_medai_video=social_medai_video,
        Travelling_Time=Travelling_Time,
        Stress_Level=Stress_Level,
        Financial_Status=Financial_Status,
        part_time_job=part_time_job,
        Prediction=val)

        return render(request, 'RUser/Predict_Campus_Behavior_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Campus_Behavior_Type.html')



