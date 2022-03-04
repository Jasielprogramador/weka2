package asdfsfasdfadsgdfghfdg;

import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.MinkowskiDistance;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
import weka.gui.PerspectiveManager.SelectedPerspectivePreferences;
import static weka.classifiers.lazy.IBk.TAGS_WEIGHTING;
import static weka.classifiers.lazy.IBk.WEIGHT_NONE;
import static weka.classifiers.lazy.IBk.WEIGHT_INVERSE;
import static weka.classifiers.lazy.IBk.WEIGHT_SIMILARITY;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import javax.swing.text.html.HTML.Tag;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;;

public class practikatzenjKNNNNNNNN {
	
	public static void main(String[] args) throws Exception {
		
		DataSource source = new DataSource(args[0]);
		Instances dataInstances = source.getDataSet();
		
		dataInstances.setClassIndex(dataInstances.numAttributes()-1);
		
		LinearNNSearch euLinearNNSearch = new LinearNNSearch();
		euLinearNNSearch.setDistanceFunction(new EuclideanDistance());
		
		LinearNNSearch maLinearNNSearch = new LinearNNSearch();
		euLinearNNSearch.setDistanceFunction(new ManhattanDistance());
		
		LinearNNSearch miLinearNNSearch = new LinearNNSearch();
		euLinearNNSearch.setDistanceFunction(new MinkowskiDistance());
		
		LinearNNSearch[] distantziaLinearNNSearch = new LinearNNSearch[] {euLinearNNSearch,maLinearNNSearch,miLinearNNSearch};
		
		SelectedTag tags[] = new SelectedTag[]{new SelectedTag(WEIGHT_NONE, TAGS_WEIGHTING),new SelectedTag(WEIGHT_INVERSE, TAGS_WEIGHTING),new SelectedTag(WEIGHT_SIMILARITY, TAGS_WEIGHTING)};
		
		double max = 0.0;
		int a = 0;
		int b = 0;
		int c = 0;
		IBk kBk= new IBk();

		
		for(int i = 0;i<dataInstances.numInstances();i++) {
			kBk.setKNN(i);
			for(int j = 0;j<distantziaLinearNNSearch.length;j++) {
				kBk.setNearestNeighbourSearchAlgorithm(distantziaLinearNNSearch[j]);
				for(int l = 0;l<tags.length;l++) {
					try {
						kBk.setDistanceWeighting(tags[l]);
						Evaluation evaluation = new Evaluation(dataInstances);
						evaluation.crossValidateModel(kBk, dataInstances, 10, new Random(1));
						if(max<evaluation.weightedFMeasure()) {
							a=i;
							b=j;
							c=l;
							max=evaluation.weightedFMeasure();
		
						}
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
			}
		}
		
		kBk=new IBk();
		kBk.setKNN(a);
		kBk.setNearestNeighbourSearchAlgorithm(distantziaLinearNNSearch[b]);
		kBk.setDistanceWeighting(tags[c]);
		
		Evaluation evaluation = new Evaluation(dataInstances);
		evaluation.crossValidateModel(kBk, dataInstances, 10, new Random(1));
		
		try {
			File file = new File(args[1]);
			FileWriter writer = new FileWriter(file);
			writer.write(null);
			writer.flush();
			writer.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
