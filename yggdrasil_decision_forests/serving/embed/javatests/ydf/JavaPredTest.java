package ydf;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import ydf.models.TestModelAbaloneRegressionGBDTV2Routing;
import ydf.models.TestModelAbaloneRegressionRFSmallRouting;
import ydf.models.TestModelAdultBinaryClassGBDTV2ClassRouting;
import ydf.models.TestModelAdultBinaryClassGBDTV2ProbaRouting;
import ydf.models.TestModelAdultBinaryClassGBDTV2ScoreRouting;
import ydf.models.TestModelAdultBinaryClassRFNWTASmallClassRouting;
import ydf.models.TestModelAdultBinaryClassRFNWTASmallProbaRouting;
import ydf.models.TestModelAdultBinaryClassRFNWTASmallScoreRouting;
import ydf.models.TestModelAdultBinaryClassRFWTASmallClassRouting;
import ydf.models.TestModelAdultBinaryClassRFWTASmallProbaRouting;
import ydf.models.TestModelAdultBinaryClassRFWTASmallScoreRouting;
import ydf.models.TestModelIrisMultiClassGBDTV2ClassRouting;
import ydf.models.TestModelIrisMultiClassGBDTV2ProbaRouting;
import ydf.models.TestModelIrisMultiClassGBDTV2ScoreRouting;
import ydf.models.TestModelIrisMultiClassRFNWTASmallClassRouting;
import ydf.models.TestModelIrisMultiClassRFNWTASmallProbaRouting;
import ydf.models.TestModelIrisMultiClassRFNWTASmallScoreRouting;
import ydf.models.TestModelIrisMultiClassRFWTASmallClassRouting;
import ydf.models.TestModelIrisMultiClassRFWTASmallProbaRouting;
import ydf.models.TestModelIrisMultiClassRFWTASmallScoreRouting;

@RunWith(JUnit4.class)
public class JavaPredTest {

  @Test
  public void testAbaloneRegressionGBDTV2Routing_knownOutput() {
    var instance =
        new TestModelAbaloneRegressionGBDTV2Routing.Instance(
            /* type= */ TestModelAbaloneRegressionGBDTV2Routing.FeatureType.M,
            /* longestshell= */ 0.455f,
            /* diameter= */ 0.365f,
            /* height= */ 0.095f,
            /* wholeweight= */ 0.514f,
            /* shuckedweight= */ 0.2245f,
            /* visceraweight= */ 0.101f,
            /* shellweight= */ 0.15f);
    float expected = 9.815921f;
    assertThat(TestModelAbaloneRegressionGBDTV2Routing.predict(instance))
        .isWithin(0.00001f)
        .of(expected);
  }

  @Test
  public void testAbaloneRegressionRFSmallRouting_knownOutput() {
    var instance =
        new TestModelAbaloneRegressionRFSmallRouting.Instance(
            /* type= */ TestModelAbaloneRegressionRFSmallRouting.FeatureType.M,
            /* longestshell= */ 0.455f,
            /* diameter= */ 0.365f,
            /* height= */ 0.095f,
            /* wholeweight= */ 0.514f,
            /* shuckedweight= */ 0.2245f,
            /* visceraweight= */ 0.101f,
            /* shellweight= */ 0.15f);
    float expected = 11.092856f;
    assertThat(TestModelAbaloneRegressionRFSmallRouting.predict(instance))
        .isWithin(0.00001f)
        .of(expected);
  }

  @Test
  public void testAdultBinaryClassGBDTV2ClassRouting_knownOutput() {

    var instance =
        new TestModelAdultBinaryClassGBDTV2ClassRouting.Instance(
            /* age= */ 39,
            /* workclass= */ TestModelAdultBinaryClassGBDTV2ClassRouting.FeatureWorkclass.STATE_GOV,
            /* fnlwgt= */ 77516,
            /* education= */ TestModelAdultBinaryClassGBDTV2ClassRouting.FeatureEducation.BACHELORS,
            /* educationNum= */ 13,
            TestModelAdultBinaryClassGBDTV2ClassRouting.FeatureMaritalStatus
                .NEVER_MARRIED, // maritalStatus
            TestModelAdultBinaryClassGBDTV2ClassRouting.FeatureOccupation
                .ADM_CLERICAL, // occupation
            TestModelAdultBinaryClassGBDTV2ClassRouting.FeatureRelationship
                .NOT_IN_FAMILY, // relationship
            /* race= */ TestModelAdultBinaryClassGBDTV2ClassRouting.FeatureRace.WHITE,
            /* sex= */ TestModelAdultBinaryClassGBDTV2ClassRouting.FeatureSex.MALE,
            /* capitalGain= */ 2174,
            /* capitalLoss= */ 0,
            /* hoursPerWeek= */ 40,
            /* nativeCountry= */ TestModelAdultBinaryClassGBDTV2ClassRouting.FeatureNativeCountry
                .UNITED_STATES);
    assertThat(TestModelAdultBinaryClassGBDTV2ClassRouting.predict(instance))
        .isEqualTo(TestModelAdultBinaryClassGBDTV2ClassRouting.Label.LT50K);
  }

  @Test
  public void testAdultBinaryClassGBDTV2ProbaRouting_knownOutput() {

    var instance =
        new TestModelAdultBinaryClassGBDTV2ProbaRouting.Instance(
            /* age= */ 39,
            /* workclass= */ TestModelAdultBinaryClassGBDTV2ProbaRouting.FeatureWorkclass.STATE_GOV,
            /* fnlwgt= */ 77516,
            /* education= */ TestModelAdultBinaryClassGBDTV2ProbaRouting.FeatureEducation.BACHELORS,
            /* educationNum= */ 13,
            TestModelAdultBinaryClassGBDTV2ProbaRouting.FeatureMaritalStatus
                .NEVER_MARRIED, // maritalStatus
            TestModelAdultBinaryClassGBDTV2ProbaRouting.FeatureOccupation
                .ADM_CLERICAL, // occupation
            TestModelAdultBinaryClassGBDTV2ProbaRouting.FeatureRelationship
                .NOT_IN_FAMILY, // relationship
            /* race= */ TestModelAdultBinaryClassGBDTV2ProbaRouting.FeatureRace.WHITE,
            /* sex= */ TestModelAdultBinaryClassGBDTV2ProbaRouting.FeatureSex.MALE,
            /* capitalGain= */ 2174,
            /* capitalLoss= */ 0,
            /* hoursPerWeek= */ 40,
            /* nativeCountry= */ TestModelAdultBinaryClassGBDTV2ProbaRouting.FeatureNativeCountry
                .UNITED_STATES);
    float expected = 0.01860435f;
    assertThat(TestModelAdultBinaryClassGBDTV2ProbaRouting.predict(instance))
        .isWithin(0.00001f)
        .of(expected);
  }

  @Test
  public void testAdultBinaryClassGBDTV2ScoreRouting_knownOutput() {

    var instance =
        new TestModelAdultBinaryClassGBDTV2ScoreRouting.Instance(
            /* age= */ 39,
            /* workclass= */ TestModelAdultBinaryClassGBDTV2ScoreRouting.FeatureWorkclass.STATE_GOV,
            /* fnlwgt= */ 77516,
            /* education= */ TestModelAdultBinaryClassGBDTV2ScoreRouting.FeatureEducation.BACHELORS,
            /* educationNum= */ 13,
            TestModelAdultBinaryClassGBDTV2ScoreRouting.FeatureMaritalStatus
                .NEVER_MARRIED, // maritalStatus
            TestModelAdultBinaryClassGBDTV2ScoreRouting.FeatureOccupation
                .ADM_CLERICAL, // occupation
            TestModelAdultBinaryClassGBDTV2ScoreRouting.FeatureRelationship
                .NOT_IN_FAMILY, // relationship
            /* race= */ TestModelAdultBinaryClassGBDTV2ScoreRouting.FeatureRace.WHITE,
            /* sex= */ TestModelAdultBinaryClassGBDTV2ScoreRouting.FeatureSex.MALE,
            /* capitalGain= */ 2174,
            /* capitalLoss= */ 0,
            /* hoursPerWeek= */ 40,
            /* nativeCountry= */ TestModelAdultBinaryClassGBDTV2ScoreRouting.FeatureNativeCountry
                .UNITED_STATES);
    float expected = -3.96557950f;
    assertThat(TestModelAdultBinaryClassGBDTV2ScoreRouting.predict(instance))
        .isWithin(0.00001f)
        .of(expected);
  }

  // Adult RF WTA

  @Test
  public void testAdultBinaryClassRFWTASmallProbaRouting_knownOutput() {

    var instance =
        new TestModelAdultBinaryClassRFWTASmallProbaRouting.Instance(
            /* age= */ 39,
            /* workclass= */ TestModelAdultBinaryClassRFWTASmallProbaRouting.FeatureWorkclass
                .STATE_GOV,
            /* fnlwgt= */ 77516,
            /* education= */ TestModelAdultBinaryClassRFWTASmallProbaRouting.FeatureEducation
                .BACHELORS,
            /* educationNum= */ 13,
            TestModelAdultBinaryClassRFWTASmallProbaRouting.FeatureMaritalStatus
                .NEVER_MARRIED, // maritalStatus
            TestModelAdultBinaryClassRFWTASmallProbaRouting.FeatureOccupation
                .ADM_CLERICAL, // occupation
            TestModelAdultBinaryClassRFWTASmallProbaRouting.FeatureRelationship
                .NOT_IN_FAMILY, // relationship
            /* race= */ TestModelAdultBinaryClassRFWTASmallProbaRouting.FeatureRace.WHITE,
            /* sex= */ TestModelAdultBinaryClassRFWTASmallProbaRouting.FeatureSex.MALE,
            /* capitalGain= */ 2174,
            /* capitalLoss= */ 0,
            /* hoursPerWeek= */ 40,
            /* nativeCountry= */ TestModelAdultBinaryClassRFWTASmallProbaRouting
                .FeatureNativeCountry.UNITED_STATES);
    float expected = 0.f;
    assertThat(TestModelAdultBinaryClassRFWTASmallProbaRouting.predict(instance))
        .isWithin(0.00001f)
        .of(expected);
  }

  @Test
  public void testAdultBinaryClassRFWTASmallClassRouting_knownOutput() {

    var instance =
        new TestModelAdultBinaryClassRFWTASmallClassRouting.Instance(
            /* age= */ 39,
            /* workclass= */ TestModelAdultBinaryClassRFWTASmallClassRouting.FeatureWorkclass
                .STATE_GOV,
            /* fnlwgt= */ 77516,
            /* education= */ TestModelAdultBinaryClassRFWTASmallClassRouting.FeatureEducation
                .BACHELORS,
            /* educationNum= */ 13,
            TestModelAdultBinaryClassRFWTASmallClassRouting.FeatureMaritalStatus
                .NEVER_MARRIED, // maritalStatus
            TestModelAdultBinaryClassRFWTASmallClassRouting.FeatureOccupation
                .ADM_CLERICAL, // occupation
            TestModelAdultBinaryClassRFWTASmallClassRouting.FeatureRelationship
                .NOT_IN_FAMILY, // relationship
            /* race= */ TestModelAdultBinaryClassRFWTASmallClassRouting.FeatureRace.WHITE,
            /* sex= */ TestModelAdultBinaryClassRFWTASmallClassRouting.FeatureSex.MALE,
            /* capitalGain= */ 2174,
            /* capitalLoss= */ 0,
            /* hoursPerWeek= */ 40,
            /* nativeCountry= */ TestModelAdultBinaryClassRFWTASmallClassRouting
                .FeatureNativeCountry.UNITED_STATES);
    assertThat(TestModelAdultBinaryClassRFWTASmallClassRouting.predict(instance))
        .isEqualTo(TestModelAdultBinaryClassRFWTASmallClassRouting.Label.LT50K);
  }

  @Test
  public void testAdultBinaryClassRFWTASmallScoreRouting_knownOutput() {

    var instance =
        new TestModelAdultBinaryClassRFWTASmallScoreRouting.Instance(
            /* age= */ 39,
            /* workclass= */ TestModelAdultBinaryClassRFWTASmallScoreRouting.FeatureWorkclass
                .STATE_GOV,
            /* fnlwgt= */ 77516,
            /* education= */ TestModelAdultBinaryClassRFWTASmallScoreRouting.FeatureEducation
                .BACHELORS,
            /* educationNum= */ 13,
            TestModelAdultBinaryClassRFWTASmallScoreRouting.FeatureMaritalStatus
                .NEVER_MARRIED, // maritalStatus
            TestModelAdultBinaryClassRFWTASmallScoreRouting.FeatureOccupation
                .ADM_CLERICAL, // occupation
            TestModelAdultBinaryClassRFWTASmallScoreRouting.FeatureRelationship
                .NOT_IN_FAMILY, // relationship
            /* race= */ TestModelAdultBinaryClassRFWTASmallScoreRouting.FeatureRace.WHITE,
            /* sex= */ TestModelAdultBinaryClassRFWTASmallScoreRouting.FeatureSex.MALE,
            /* capitalGain= */ 2174,
            /* capitalLoss= */ 0,
            /* hoursPerWeek= */ 40,
            /* nativeCountry= */ TestModelAdultBinaryClassRFWTASmallScoreRouting
                .FeatureNativeCountry.UNITED_STATES);
    byte expected = 0;
    assertThat(TestModelAdultBinaryClassRFWTASmallScoreRouting.predict(instance))
        .isEqualTo(expected);
  }

  // Adult RF NWTA

  @Test
  public void testAdultBinaryClassRFNWTASmallProbaRouting_knownOutput() {

    var instance =
        new TestModelAdultBinaryClassRFNWTASmallProbaRouting.Instance(
            /* age= */ 39,
            /* workclass= */ TestModelAdultBinaryClassRFNWTASmallProbaRouting.FeatureWorkclass
                .STATE_GOV,
            /* fnlwgt= */ 77516,
            /* education= */ TestModelAdultBinaryClassRFNWTASmallProbaRouting.FeatureEducation
                .BACHELORS,
            /* educationNum= */ 13,
            TestModelAdultBinaryClassRFNWTASmallProbaRouting.FeatureMaritalStatus
                .NEVER_MARRIED, // maritalStatus
            TestModelAdultBinaryClassRFNWTASmallProbaRouting.FeatureOccupation
                .ADM_CLERICAL, // occupation
            TestModelAdultBinaryClassRFNWTASmallProbaRouting.FeatureRelationship
                .NOT_IN_FAMILY, // relationship
            /* race= */ TestModelAdultBinaryClassRFNWTASmallProbaRouting.FeatureRace.WHITE,
            /* sex= */ TestModelAdultBinaryClassRFNWTASmallProbaRouting.FeatureSex.MALE,
            /* capitalGain= */ 2174,
            /* capitalLoss= */ 0,
            /* hoursPerWeek= */ 40,
            /* nativeCountry= */ TestModelAdultBinaryClassRFNWTASmallProbaRouting
                .FeatureNativeCountry.UNITED_STATES);
    float expected = 0.01538462f;
    assertThat(TestModelAdultBinaryClassRFNWTASmallProbaRouting.predict(instance))
        .isWithin(0.00001f)
        .of(expected);
  }

  @Test
  public void testAdultBinaryClassRFNWTASmallClassRouting_knownOutput() {

    var instance =
        new TestModelAdultBinaryClassRFNWTASmallClassRouting.Instance(
            /* age= */ 39,
            /* workclass= */ TestModelAdultBinaryClassRFNWTASmallClassRouting.FeatureWorkclass
                .STATE_GOV,
            /* fnlwgt= */ 77516,
            /* education= */ TestModelAdultBinaryClassRFNWTASmallClassRouting.FeatureEducation
                .BACHELORS,
            /* educationNum= */ 13,
            TestModelAdultBinaryClassRFNWTASmallClassRouting.FeatureMaritalStatus
                .NEVER_MARRIED, // maritalStatus
            TestModelAdultBinaryClassRFNWTASmallClassRouting.FeatureOccupation
                .ADM_CLERICAL, // occupation
            TestModelAdultBinaryClassRFNWTASmallClassRouting.FeatureRelationship
                .NOT_IN_FAMILY, // relationship
            /* race= */ TestModelAdultBinaryClassRFNWTASmallClassRouting.FeatureRace.WHITE,
            /* sex= */ TestModelAdultBinaryClassRFNWTASmallClassRouting.FeatureSex.MALE,
            /* capitalGain= */ 2174,
            /* capitalLoss= */ 0,
            /* hoursPerWeek= */ 40,
            /* nativeCountry= */ TestModelAdultBinaryClassRFNWTASmallClassRouting
                .FeatureNativeCountry.UNITED_STATES);
    assertThat(TestModelAdultBinaryClassRFNWTASmallClassRouting.predict(instance))
        .isEqualTo(TestModelAdultBinaryClassRFNWTASmallClassRouting.Label.LT50K);
  }

  @Test
  public void testAdultBinaryClassRFNWTASmallScoreRouting_knownOutput() {

    var instance =
        new TestModelAdultBinaryClassRFNWTASmallScoreRouting.Instance(
            /* age= */ 39,
            /* workclass= */ TestModelAdultBinaryClassRFNWTASmallScoreRouting.FeatureWorkclass
                .STATE_GOV,
            /* fnlwgt= */ 77516,
            /* education= */ TestModelAdultBinaryClassRFNWTASmallScoreRouting.FeatureEducation
                .BACHELORS,
            /* educationNum= */ 13,
            TestModelAdultBinaryClassRFNWTASmallScoreRouting.FeatureMaritalStatus
                .NEVER_MARRIED, // maritalStatus
            TestModelAdultBinaryClassRFNWTASmallScoreRouting.FeatureOccupation
                .ADM_CLERICAL, // occupation
            TestModelAdultBinaryClassRFNWTASmallScoreRouting.FeatureRelationship
                .NOT_IN_FAMILY, // relationship
            /* race= */ TestModelAdultBinaryClassRFNWTASmallScoreRouting.FeatureRace.WHITE,
            /* sex= */ TestModelAdultBinaryClassRFNWTASmallScoreRouting.FeatureSex.MALE,
            /* capitalGain= */ 2174,
            /* capitalLoss= */ 0,
            /* hoursPerWeek= */ 40,
            /* nativeCountry= */ TestModelAdultBinaryClassRFNWTASmallScoreRouting
                .FeatureNativeCountry.UNITED_STATES);
    float expected = 0.01538462f;
    assertThat(TestModelAdultBinaryClassRFNWTASmallScoreRouting.predict(instance))
        .isWithin(0.00001f)
        .of(expected);
  }

  // Iris GBDT

  @Test
  public void testIrisMultiClassGBDTV2ClassRouting_knownOutput() {
    var instance =
        new TestModelIrisMultiClassGBDTV2ClassRouting.Instance(
            /* sepal_Length= */ 5.1f,
            /* sepal_Width= */ 3.5f,
            /* petal_Length= */ 1.4f,
            /* petal_Width= */ 0.2f);
    assertThat(TestModelIrisMultiClassGBDTV2ClassRouting.predict(instance))
        .isEqualTo(TestModelIrisMultiClassGBDTV2ClassRouting.Label.SETOSA);
  }

  @Test
  public void testIrisMultiClassGBDTV2ProbaRouting_knownOutput() {
    var instance =
        new TestModelIrisMultiClassGBDTV2ProbaRouting.Instance(
            /* sepal_Length= */ 5.1f,
            /* sepal_Width= */ 3.5f,
            /* petal_Length= */ 1.4f,
            /* petal_Width= */ 0.2f);
    float[] proba = TestModelIrisMultiClassGBDTV2ProbaRouting.predict(instance);
    assertThat(proba).hasLength(3);
    assertThat(
            proba[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.SETOSA)])
        .isWithin(0.00001f)
        .of(0.9789308f);
    assertThat(
            proba[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.VERSICOLOR)])
        .isWithin(0.00001f)
        .of(0.01048146f);
    assertThat(
            proba[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.VIRGINICA)])
        .isWithin(0.00001f)
        .of(0.01058776f);
  }

  @Test
  public void testIrisMultiClassGBDTV2ScoreRouting_knownOutput() {
    var instance =
        new TestModelIrisMultiClassGBDTV2ScoreRouting.Instance(
            /* sepal_Length= */ 5.1f,
            /* sepal_Width= */ 3.5f,
            /* petal_Length= */ 1.4f,
            /* petal_Width= */ 0.2f);
    float[] scores = TestModelIrisMultiClassGBDTV2ScoreRouting.predict(instance);
    assertThat(scores).hasLength(3);
    assertThat(
            scores[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.SETOSA)])
        .isWithin(0.00001f)
        .of(2.497073f);
    assertThat(
            scores[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.VERSICOLOR)])
        .isWithin(0.00001f)
        .of(-2.0397801f);
    assertThat(
            scores[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.VIRGINICA)])
        .isWithin(0.00001f)
        .of(-2.029691f);
  }

  // Iris RF WTA

  @Test
  public void testIrisMultiClassRFWTASmallClassRouting_knownOutput() {
    var instance =
        new TestModelIrisMultiClassRFWTASmallClassRouting.Instance(
            /* sepal_Length= */ 5.1f,
            /* sepal_Width= */ 3.5f,
            /* petal_Length= */ 1.4f,
            /* petal_Width= */ 0.2f);
    assertThat(TestModelIrisMultiClassRFWTASmallClassRouting.predict(instance))
        .isEqualTo(TestModelIrisMultiClassRFWTASmallClassRouting.Label.SETOSA);
  }

  @Test
  public void testIrisMultiClassRFWTASmallScoreRouting_knownOutput() {
    var instance =
        new TestModelIrisMultiClassRFWTASmallScoreRouting.Instance(
            /* sepal_Length= */ 5.1f,
            /* sepal_Width= */ 3.5f,
            /* petal_Length= */ 1.4f,
            /* petal_Width= */ 0.2f);
    byte[] scores = TestModelIrisMultiClassRFWTASmallScoreRouting.predict(instance);
    assertThat(scores).hasLength(3);
    assertThat(
            scores[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.SETOSA)])
        .isEqualTo((byte) 10);
    assertThat(
            scores[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.VERSICOLOR)])
        .isEqualTo((byte) 0);
    assertThat(
            scores[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.VIRGINICA)])
        .isEqualTo((byte) 0);
  }

  @Test
  public void testIrisMultiClassRFWTASmallProbaRouting_knownOutput() {
    var instance =
        new TestModelIrisMultiClassRFWTASmallProbaRouting.Instance(
            /* sepal_Length= */ 5.1f,
            /* sepal_Width= */ 3.5f,
            /* petal_Length= */ 1.4f,
            /* petal_Width= */ 0.2f);
    float[] proba = TestModelIrisMultiClassRFWTASmallProbaRouting.predict(instance);
    assertThat(proba).hasLength(3);
    assertThat(
            proba[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.SETOSA)])
        .isEqualTo(1.f);
    assertThat(
            proba[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.VERSICOLOR)])
        .isEqualTo(0.f);
    assertThat(
            proba[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.VIRGINICA)])
        .isEqualTo(0.f);
  }

  // Iris RF NWTA

  @Test
  public void testIrisMultiClassRFNWTASmallClassRouting_knownOutput() {
    var instance =
        new TestModelIrisMultiClassRFNWTASmallClassRouting.Instance(
            /* sepal_Length= */ 5.1f,
            /* sepal_Width= */ 3.5f,
            /* petal_Length= */ 1.4f,
            /* petal_Width= */ 0.2f);
    assertThat(TestModelIrisMultiClassRFNWTASmallClassRouting.predict(instance))
        .isEqualTo(TestModelIrisMultiClassRFNWTASmallClassRouting.Label.SETOSA);
  }

  @Test
  public void testIrisMultiClassRFNWTASmallScoreRouting_knownOutput() {
    var instance =
        new TestModelIrisMultiClassRFNWTASmallScoreRouting.Instance(
            /* sepal_Length= */ 5.1f,
            /* sepal_Width= */ 3.5f,
            /* petal_Length= */ 1.4f,
            /* petal_Width= */ 0.2f);
    float[] scores = TestModelIrisMultiClassRFNWTASmallScoreRouting.predict(instance);
    assertThat(scores).hasLength(3);
    assertThat(
            scores[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.SETOSA)])
        .isWithin(0.00001f)
        .of(1.f);
    assertThat(
            scores[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.VERSICOLOR)])
        .isWithin(0.00001f)
        .of(0.f);
    assertThat(
            scores[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.VIRGINICA)])
        .isWithin(0.00001f)
        .of(0.f);
  }

  @Test
  public void testIrisMultiClassRFNWTASmallProbaRouting_knownOutput() {
    var instance =
        new TestModelIrisMultiClassRFNWTASmallProbaRouting.Instance(
            /* sepal_Length= */ 5.1f,
            /* sepal_Width= */ 3.5f,
            /* petal_Length= */ 1.4f,
            /* petal_Width= */ 0.2f);
    float[] proba = TestModelIrisMultiClassRFNWTASmallProbaRouting.predict(instance);
    assertThat(proba).hasLength(3);
    assertThat(
            proba[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.SETOSA)])
        .isWithin(0.00001f)
        .of(1.f);
    assertThat(
            proba[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.VERSICOLOR)])
        .isWithin(0.00001f)
        .of(0.f);
    assertThat(
            proba[
                TestModelIrisMultiClassGBDTV2ProbaRouting.getLabelIndex(
                    TestModelIrisMultiClassGBDTV2ProbaRouting.Label.VIRGINICA)])
        .isWithin(0.00001f)
        .of(0.f);
  }

  //
  // NA Values
  //

  @Test
  public void testAbaloneRegressionGBDTV2Routing_withNa() {
    var instance =
        new TestModelAbaloneRegressionGBDTV2Routing.Instance(
            /* type= */ TestModelAbaloneRegressionGBDTV2Routing.FeatureType.M,
            /* longestshell= */ Float.NaN,
            /* diameter= */ 0.365f,
            /* height= */ Float.NaN,
            /* wholeweight= */ 0.514f,
            /* shuckedweight= */ Float.NaN,
            /* visceraweight= */ Float.NaN,
            /* shellweight= */ Float.NaN);
    float expected = 9.362932f;
    assertThat(TestModelAbaloneRegressionGBDTV2Routing.predict(instance))
        .isWithin(0.00001f)
        .of(expected);
  }
}
