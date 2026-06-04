\# INT332 DevOps – Scenario Based CheatSheet (Commands + Solutions)



\---



\# Q1. Docker Image Creation (Flask App)



FROM python:3.9

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD \["python", "app.py"]



docker build -t flask-app:v1 .

docker run -d -p 5000:5000 flask-app:v1



\---



\# Q2. Docker Networking + MySQL



docker network create backend-net



docker run -d --name mysql --network backend-net -e MYSQL\_ROOT\_PASSWORD=root mysql



docker run -d --name api --network backend-net -p 8080:8080 myapi



docker exec -it api ping mysql



\---



\# Q3. Docker Volumes (PostgreSQL)



docker volume create pgdata



docker run -d \\

&#x20; --name postgres \\

&#x20; -v pgdata:/var/lib/postgresql/data \\

&#x20; -e POSTGRES\_PASSWORD=pass postgres



docker stop postgres

docker rm postgres



docker run -d \\

&#x20; --name postgres2 \\

&#x20; -v pgdata:/var/lib/postgresql/data \\

&#x20; -e POSTGRES\_PASSWORD=pass postgres



\---



\# Q4. Multi-Stage Dockerfile (Java)



FROM maven:3.8 AS build

WORKDIR /app

COPY . .

RUN mvn clean package



FROM openjdk:17

WORKDIR /app

COPY --from=build /app/target/app.jar app.jar

CMD \["java","-jar","app.jar"]



\---



\# Q5. Docker Compose (Node + Mongo + Nginx)



version: "3.8"



services:

&#x20; backend:

&#x20;   image: node

&#x20;   ports:

&#x20;     - "3000:3000"



&#x20; mongo:

&#x20;   image: mongo



&#x20; frontend:

&#x20;   image: nginx

&#x20;   ports:

&#x20;     - "80:80"



\---



\# Q6. GHCR Push



docker login ghcr.io -u USERNAME -p TOKEN



docker tag myapp ghcr.io/username/myapp:v1



docker push ghcr.io/username/myapp:v1



docker pull ghcr.io/username/myapp:v1



\---



\# Q7. Maven Docker Plugin (POM)



<plugin>

&#x20; <groupId>com.spotify</groupId>

&#x20; <artifactId>dockerfile-maven-plugin</artifactId>

&#x20; <version>1.4.13</version>

</plugin>



mvn clean package

docker images



\---



\# Q8. GitHub Actions Docker CI



name: CI



on: push



jobs:

&#x20; build:

&#x20;   runs-on: ubuntu-latest



&#x20;   steps:

&#x20;     - uses: actions/checkout@v4



&#x20;     - name: Build Image

&#x20;       run: docker build -t myapp .



&#x20;     - name: Login

&#x20;       run: echo ${{ secrets.DOCKER\_PASS }} | docker login -u USER --password-stdin



&#x20;     - name: Push

&#x20;       run: |

&#x20;         docker tag myapp user/myapp:v1

&#x20;         docker push user/myapp:v1



\---



\# Q9. Jenkins Pipeline (Maven + Docker)



pipeline {

&#x20; agent any



&#x20; stages {

&#x20;   stage('Checkout') {

&#x20;     steps { git 'repo-url' }

&#x20;   }



&#x20;   stage('Build') {

&#x20;     steps { sh 'mvn clean package' }

&#x20;   }



&#x20;   stage('Docker Build') {

&#x20;     steps { sh 'docker build -t app .' }

&#x20;   }



&#x20;   stage('Push') {

&#x20;     steps { sh 'docker push app' }

&#x20;   }

&#x20; }

}



\---



\# Q10. Jenkins CI/CD Deployment



ssh user@server "docker pull app \&\& docker run -d -p 80:80 app"



\---



\# Q11. Docker Layer Optimization



COPY package.json .

RUN npm install

COPY . .



\---



\# Q12. Namespace + cgroups



docker run -m 256m --cpus="0.5" nginx



docker stats



\---



\# Q13. ENV + ENTRYPOINT + CMD



FROM ubuntu

ENV URL=https://example.com

ENTRYPOINT \["curl"]

CMD \["https://default.com"]



docker run image https://newurl.com



\---



\# Q14. Overlay Network



docker network create -d overlay mynet



docker service create --name web --replicas 2 nginx



\---



\# Q15. WordPress + MySQL



version: "3.8"



services:

&#x20; db:

&#x20;   image: mysql

&#x20;   environment:

&#x20;     MYSQL\_ROOT\_PASSWORD: root



&#x20; wordpress:

&#x20;   image: wordpress

&#x20;   ports:

&#x20;     - "80:80"



\---



\# Q16. Docker Swarm



docker swarm init



docker service create --replicas 3 -p 80:80 nginx



docker service scale web=5



\---



\# Q17. Maven Dependency Fix



mvn dependency:tree



<dependencyManagement>

&#x20; <dependencies>

&#x20;   <dependency>

&#x20;     <groupId>x</groupId>

&#x20;     <artifactId>y</artifactId>

&#x20;     <version>1.0</version>

&#x20;   </dependency>

&#x20; </dependencies>

</dependencyManagement>



\---



\# Q18. GitHub Actions Matrix



strategy:

&#x20; matrix:

&#x20;   java: \[11, 17, 21]



\---



\# Q19. Jenkins Docker Agent



pipeline {

&#x20; agent {

&#x20;   docker { image 'maven:3.8' }

&#x20; }

}



\---



\# Q20. Jenkins RBAC + Artifacts



archiveArtifacts artifacts: '\*\*/\*.jar'



properties(\[

&#x20; buildDiscarder(logRotator(numToKeepStr: '5'))

])

