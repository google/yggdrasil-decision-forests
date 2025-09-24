package ydf;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import ydf.models.TestModelAbaloneRegressionGBDTV2Routing;
import ydf.models.TestModelAdultBinaryClassGBDTV2ClassRouting;
import ydf.models.TestModelAdultBinaryClassGBDTV2ProbaRouting;
import ydf.models.TestModelIrisMultiClassGBDTV2ProbaRouting;

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
  public void testIrisMultiClassGBDTV2ProbaRouting_knownOutput() {
    var instance =
        new TestModelIrisMultiClassGBDTV2ProbaRouting.Instance(
            /* sepal_Length= */ 5.1f,
            /* sepal_Width= */ 3.5f,
            /* petal_Length= */ 1.4f,
            /* petal_Width= */ 0.2f);
    assertThat(TestModelIrisMultiClassGBDTV2ProbaRouting.predict(instance))
        .isEqualTo(TestModelIrisMultiClassGBDTV2ProbaRouting.Label.SETOSA);
  }
}
