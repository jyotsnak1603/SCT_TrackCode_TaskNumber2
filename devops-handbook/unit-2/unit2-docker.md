\# UNIT 2: Dockerfile, Image Building \& Container Management



\---



\# 1. Docker Image Build Commands



docker build -t myapp .

docker build -t myapp:v1 .



docker images

docker rmi myapp

docker history myapp



\---



\# 2. Run Container from Image



docker run myapp

docker run -d myapp



docker run -d -p 8080:80 myapp



docker ps

docker ps -a



docker stop <container\_id>

docker start <container\_id>

docker rm <container\_id>



\---



\# 3. Dockerfile Example (Java App)



FROM openjdk:17



WORKDIR /app



COPY . .



RUN javac Main.java



CMD \["java", "Main"]



\---



\# 4. Dockerfile Example (Node App)



FROM node:18



WORKDIR /app



COPY package.json .



RUN npm install



COPY . .



CMD \["node", "app.js"]



\---



\# 5. Dockerfile Instructions



FROM → Base image

WORKDIR → Working directory inside container

COPY → Copy files from host to container

ADD → Similar to COPY but supports URLs \& tar extraction

RUN → Executes commands during build

CMD → Default command at container runtime

ENTRYPOINT → Fixed execution command



\---



\# 6. Image Tagging \& Versioning



docker build -t myapp:latest .

docker build -t myapp:v1 .

docker tag myapp:latest myapp:v2



\---



\# 7. Inspect Image Layers



docker history myapp

docker inspect myapp



\---



\# 8. Copy-on-Write Concept (Commands Related View)



docker run -d ubuntu

docker exec -it <container\_id> bash

