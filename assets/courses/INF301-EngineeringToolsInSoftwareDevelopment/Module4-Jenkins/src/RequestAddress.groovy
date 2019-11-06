def slurpJSON(myString) {
  return new groovy.json.JsonSlurperClassic().parseText(myString);
}

node {
   def mvnHome
   stage('Send request') { // for display purposes
      // Get some code from a GitHub repository
      query = 'https://api-adresse.data.gouv.fr/search/?q=' + java.net.URLEncoder.encode(ADDRESS, "UTF-8");
      resultAsString = sh(returnStdout: true, script: 'curl -XGET "' + query + '"');
      echo resultAsString
   }
   stage('Process data') {
      def resultAsMap = slurpJSON(resultAsString);
      longitude = 0;
      latitude = 0;
      postcode = '';
      if ("features" in resultAsMap) {
        firstResult = resultAsMap["features"][0];
        if ("geometry" in firstResult) {
          longitude = firstResult["geometry"]["coordinates"][0];
          latitude = firstResult["geometry"]["coordinates"][1];
        }
        if ("properties" in firstResult) {
          postcode = firstResult["properties"]["postcode"];
        }
      }
   }
   stage('Display result') {
      echo "Longitude: " + longitude
      echo "Latitude: " + latitude
      echo "Postcode: " + postcode 
   }
}