import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.4.10"
}
group = "com.musaabshirgar1"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven(url = "https://jitpack.io")
}
dependencies {
    implementation("org.jetbrains.kotlin:kotlin-stdlib:1.4.10")
    implementation("org.jetbrains.kotlin:kotlin-reflect:1.1.0")
    implementation("org.ojalgo:ojalgo:47.2.0")
    implementation("no.tornado:tornadofx:1.7.20")
    implementation("org.deeplearning4j:deeplearning4j-core:1.0.0-beta2")
    implementation("org.nd4j:nd4j-native-platform:1.0.0-beta2")
    implementation("org.nield:kotlin-statistics:1.2.1")
    testImplementation(kotlin("test-junit"))
}
tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}