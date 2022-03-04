package asdfsfasdfadsgdfghfdg;

import java.io.FileWriter;
import java.io.PrintWriter;

import javax.lang.model.element.NestingKind;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class otrooooooo {

	public static void main(String[] args) throws Exception {
		//DataSource source=new DataSource("/home/lsi/Escritorio/Datuak-20210305/data_supervised.arff");
		DataSource source=new DataSource(args[0]);
		Instances data=source.getDataSet();
		
		data.setClassIndex(data.numAttributes()-1);
		
		SMO smo = new SMO();
		PolyKernel kernel = new PolyKernel();
		
		double max = 0.0;
		double exponent = 0.0;
		
		for(int i =0;i<3;i++) {
			kernel.setExponent(i);
			smo.setKernel(kernel);
			Evaluation evaluation = new Evaluation(data);
			evaluation.crossValidateModel(smo, data, i, null);
			
			if(max<evaluation.weightedFMeasure()) {
				max = evaluation.weightedFMeasure();
				exponent=i;
			}
			
		}
		kernel.setExponent(exponent);
		smo.setKernel(kernel);
		smo.buildClassifier(data);
		SerializationHelper.write(args[3], smo);
		Classifier aClassifier = (Classifier) SerializationHelper.read(args[2]);
		Evaluation evaluation = new Evaluation(data);
		evaluation.crossValidateModel(aClassifier, data, 0, null);
		
		FileWriter filePre = new FileWriter(args[4]);
		PrintWriter pwPre = new PrintWriter(filePre);
		
		int i = 0;
		for(Prediction p :evaluation.predictions()) {
			String iragarpenaString = data.attribute(data.classIndex()).value((int) p.predicted());
			String erreString = data.attribute(data.classIndex()).value((int) p.predicted());
			
			String erroString = "";
			if(Double.isNaN(p.actual())) {
				iragarpenaString="?";
			}
			if(iragarpenaString!=erreString) {
				erroString="$";
			}
			else {
				erroString="-";
			}
		}
		
	}
		
}
