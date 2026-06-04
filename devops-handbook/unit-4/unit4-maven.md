\# UNIT 4: Maven Build Automation



\---



\# 1. Maven Project Structure



src/

&#x20;├── main/

&#x20;├── test/

pom.xml



\---



\# 2. Basic pom.xml Example



<project xmlns="http://maven.apache.org/POM/4.0.0">



&#x20; <modelVersion>4.0.0</modelVersion>



&#x20; <groupId>com.dev</groupId>

&#x20; <artifactId>demo-app</artifactId>

&#x20; <version>1.0</version>



</project>



\---



\# 3. Maven Build Lifecycle



validate

compile

test

package

verify

install

deploy



\---



\# 4. Maven Commands



mvn -version

mvn clean

mvn compile

mvn test

mvn package

mvn install



\---



\# 5. Dependencies Example (JUnit)



<dependencies>



&#x20; <dependency>

&#x20;   <groupId>junit</groupId>

&#x20;   <artifactId>junit</artifactId>

&#x20;   <version>4.13.2</version>

&#x20;   <scope>test</scope>

&#x20; </dependency>



</dependencies>



\---



\# 6. Maven Plugins



maven-compiler-plugin → compile code

maven-surefire-plugin → run tests

maven-shade-plugin → create fat JAR



\---



\# 7. Dependency Scope



compile → default (available everywhere)

test → only testing

provided → provided by runtime

runtime → needed during execution



\---



\# 8. Transitive Dependencies



If A depends on B and B depends on C → C is automatically included



\---



\# 9. Maven + Docker Integration (Commands View)



mvn clean package



docker build -t java-app .

docker run -d -p 8080:8080 java-app



\---



\# 10. Key Exam Points



\- pom.xml = project configuration

\- Maven automates build lifecycle

\- handles dependencies automatically

\- plugins extend functionality

\- mvn clean package is most important command

