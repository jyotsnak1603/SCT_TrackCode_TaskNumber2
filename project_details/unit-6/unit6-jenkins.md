\# UNIT 6: CI/CD with Jenkins



\---



\# 1. Jenkins Architecture (Master–Agent Model)



Jenkins follows distributed architecture:



\- Master → controls scheduling, UI, pipelines

\- Agent → executes jobs (build, test, deploy)



Real-world analogy:

Master = Manager

Agent = Workers



\---



\# 2. Jenkins Job Types



\- Freestyle Project → GUI-based jobs

\- Pipeline Project → Code-based CI/CD (Jenkinsfile)

\- Multibranch Pipeline → Auto detects branches



\---



\# 3. Jenkins Pipeline Structure (Jenkinsfile)



pipeline {

&#x20;   agent any



&#x20;   stages {



&#x20;       stage('Checkout') {

&#x20;           steps {

&#x20;               git 'https://github.com/jyotsnak1603/SCT\_TrackCode\_TaskNumber2.git'

&#x20;           }

&#x20;       }



&#x20;       stage('Build') {

&#x20;           steps {

&#x20;               sh 'mvn clean package'

&#x20;           }

&#x20;       }



&#x20;       stage('Test') {

&#x20;           steps {

&#x20;               sh 'mvn test'

&#x20;           }

&#x20;       }



&#x20;   }



&#x20;   post {

&#x20;       success {

&#x20;           echo 'Build Successful'

&#x20;       }

&#x20;       failure {

&#x20;           echo 'Build Failed'

&#x20;       }

&#x20;   }

}



\---



\# 4. Jenkins Pipeline Types



Declarative Pipeline → structured and recommended

Scripted Pipeline → flexible but complex



\---



\# 5. Jenkins + Maven Integration



Steps:

1\. Configure Maven in Jenkins Global Tools

2\. Use mvn commands in pipeline



Commands:

mvn clean

mvn compile

mvn test

mvn package



\---



\# 6. Jenkins + Docker Integration



docker build -t myapp .

docker run -d -p 8080:8080 myapp

docker push myapp



\---



\# 7. Build Triggers



\- Poll SCM

\- Webhooks (GitHub trigger)

\- Manual build



\---



\# 8. Post Build Actions



\- Archive artifacts

\- Send email notifications

\- Deploy application

\- Show test reports



\---



\# 9. Jenkins Agents (Execution Nodes)



Types:

\- SSH agents

\- Docker-based agents

\- Kubernetes agents



\---



\# 10. CI/CD Flow



Code → GitHub → Jenkins → Build → Test → Package → Deploy



\---



\# 11. Key Exam Points



\- Jenkins automates CI/CD pipelines

\- Master manages, Agent executes

\- Jenkinsfile defines pipeline as code

\- integrates with Maven + Docker + GitHub

\- post section handles success/failure actions

