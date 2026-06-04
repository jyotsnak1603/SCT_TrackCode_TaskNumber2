\# UNIT 5: Continuous Integration (CI) with GitHub Actions



\---



\# 1. What is GitHub Actions Workflow?



A workflow is an automated pipeline defined using a YAML file inside:



.github/workflows/



\---



\# 2. Basic Workflow Structure



name: CI Pipeline



on:

&#x20; push:

&#x20;   branches: \["main"]



jobs:

&#x20; build:

&#x20;   runs-on: ubuntu-latest



&#x20;   steps:

&#x20;     - name: Checkout Code

&#x20;       uses: actions/checkout@v4



&#x20;     - name: Setup Java

&#x20;       uses: actions/setup-java@v4

&#x20;       with:

&#x20;         java-version: '17'



&#x20;     - name: Build Project

&#x20;       run: mvn clean install



\---



\# 3. Key Components



Workflow → entire automation process

Jobs → group of steps

Steps → individual tasks

Actions → reusable commands

Runner → machine that executes jobs



\---



\# 4. Workflow Triggers



on:

&#x20; push

&#x20; pull\_request

&#x20; schedule



\---



\# 5. Multi-Job Example



jobs:

&#x20; build:

&#x20; test:

&#x20; deploy:



\---



\# 6. Matrix Strategy Example



strategy:

&#x20; matrix:

&#x20;   java: \[8, 11, 17]



\---



\# 7. Caching Dependencies



\- name: Cache Maven

&#x20; uses: actions/cache@v4

&#x20; with:

&#x20;   path: \~/.m2

&#x20;   key: maven-cache



\---



\# 8. GitHub Actions + Docker Commands



docker build -t myapp .

docker run -d -p 8080:8080 myapp



\---



\# 9. Deployment Example (Concept)



\- build → compile code

\- test → run unit tests

\- package → create artifact

\- deploy → push to server/cloud



\---



\# 10. Key Exam Points



\- YAML file defines workflow

\- jobs run on GitHub-hosted runners

\- triggers define automation events

\- actions are reusable modules

\- CI = automatic build + test process

