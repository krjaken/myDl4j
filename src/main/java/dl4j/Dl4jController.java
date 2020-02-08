package dl4j;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.api.transform.ui.HtmlAnalysis;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.AnalyzeLocal;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.EvaluationAveraging;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Dl4jController {

    private int batchSize = 80;
    private String modelPath = "C:/java/myDl4j/src/main/resources/source/model.bin";
    private String path = "C:/java/myDl4j/src/main/resources/source/Churn_Modelling.csv";
    private String pathHtmlAnalise = "C:/java/myDl4j/src/main/resources/source/analysis.html";
    private TransformProcess transformProcess;
    private FileSplit inputSplit;
    private Schema schema;
    private CSVRecordReader recordReader;
    private DataAnalysis analysis;
    private Schema finalSchema;
    private RecordReaderDataSetIterator trainIterator;
    private MultiLayerConfiguration config;
    private MultiLayerNetwork model;


    public Dl4jController() {
        Random random = new Random();
        random.setSeed(0xC0FFEE);
        inputSplit = new FileSplit(new File(path));//, random);

        recordReader = new CSVRecordReader(1);
        try {
            recordReader.initialize(inputSplit);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        initSchema();
        analiseData();
        finaliseSchema();
        dataAnalise();
        createModel();
        learn();
        checkLearning();
        saveMadel();
        useModel();
    }

    private void useModel() {
        File modelSave = new File(modelPath);
        try {
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelSave);
        } catch (IOException e) {
            e.printStackTrace();
        }
        DataAnalysis analysis = DataAnalysis.fromJson(ModelSerializer.getObjectFromFile(modelSave, "dataanalysis"));
        Schema targetSchema = Schema.fromJson(ModelSerializer.getObjectFromFile(modelSave, "schema"));

        List rawData = Arrays.asList(26, 8, 1, 547, 97460.1, 43093.67, "France", "Male", "1", "1");

        Schema schema = new Schema.Builder()
                .addColumnsInteger("Age", "Tenure", "Num Of Products", "Credit Score")
                .addColumnsDouble("Balance", "Estimated Salary")
                .addColumnCategorical("Geography", "France", "Germany", "Spain")
                .addColumnCategorical("Gender", "Female", "Male")
                .addColumnCategorical("Has Credit Card", "0", "1")
                .addColumnCategorical("Is Active Member", "0", "1")
                .build();

        String[] newOrder = targetSchema.getColumnNames().stream().filter(it -> !it.equals("Exited")).toArray(String[]::new);

        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .categoricalToOneHot("Geography", "Gender", "Has Credit Card", "Is Active Member")
                .integerToOneHot("Num Of Products", 1, 4)
                .normalize("Tenure", Normalize.MinMax, analysis)
                .normalize("Age", Normalize.Standardize, analysis)
                .normalize("Credit Score", Normalize.Log2Mean, analysis)
                .normalize("Balance", Normalize.Log2MeanExcludingMin, analysis)
                .normalize("Estimated Salary", Normalize.Log2MeanExcludingMin, analysis)
                .reorderColumns(newOrder)
                .build();

        List<Writable> record = RecordConverter.toRecord(schema, rawData);
        List<Writable> transformed = transformProcess.execute(record);
        INDArray data = RecordConverter.toArray(transformed);

        int[] labelIndices = model.predict(data); // = [0] = Will stay a customer
        


    }

    private void saveMadel() {
        File modelSave = new File(modelPath);
        try {
            ModelSerializer.addObjectToFile(modelSave, "dataanalysis", analysis.toJson());
            ModelSerializer.addObjectToFile(modelSave, "schema", finalSchema.toJson());
            model.save(modelSave);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void checkLearning() {
        TransformProcessRecordReader testRecordReader = new TransformProcessRecordReader(new CSVRecordReader(1), transformProcess);
        try {
            testRecordReader.initialize(new FileSplit(new File(path)));
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        RecordReaderDataSetIterator testIterator = new RecordReaderDataSetIterator.Builder(testRecordReader, batchSize)
                .classification(finalSchema.getIndexOfColumn("Exited"), 2)
                .build();

        Evaluation evaluate = model.evaluate(testIterator);
        System.out.println(evaluate.stats());
        System.out.println("MCC: " + evaluate.matthewsCorrelation(EvaluationAveraging.Macro));

    }

    private void learn() {
        model = new MultiLayerNetwork(config);
        model.addListeners(new ScoreIterationListener(50));
        //addStatisticModelListener();
        model.init();
        model.fit(trainIterator, 59);
    }

    private void addStatisticModelListener() {
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.addListeners(new StatsListener(statsStorage, 50));
    }

    private void createModel() {
        config = new NeuralNetConfiguration.Builder()
                .seed(0xC0FFEE)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.TANH)
                .updater(new Adam.Builder().learningRate(0.001).build())
                .l2(0.0000316)
                .list(
                        new DenseLayer.Builder().nOut(25).build(),
                        new DenseLayer.Builder().nOut(25).build(),
                        new DenseLayer.Builder().nOut(25).build(),
                        new DenseLayer.Builder().nOut(25).build(),
                        new DenseLayer.Builder().nOut(25).build(),
                        new OutputLayer.Builder(new LossMCXENT()).nOut(2).activation(Activation.SOFTMAX).build()
                )
                .setInputType(InputType.feedForward(finalSchema.numColumns() - 1))
                .build();
    }

    private void dataAnalise() {
        TransformProcessRecordReader trainRecordReader = new TransformProcessRecordReader(new CSVRecordReader(1), transformProcess);
        try {
            trainRecordReader.initialize(inputSplit);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        trainIterator = new RecordReaderDataSetIterator.Builder(trainRecordReader, batchSize)
                .classification(finalSchema.getIndexOfColumn("Exited"), 2)
                .build();
    }

    private void finaliseSchema() {
        transformProcess = new TransformProcess.Builder(schema)
                .removeColumns("Row Number", "Customer Id", "Surname")
                .categoricalToOneHot("Geography", "Gender", "Has Credit Card", "Is Active Member")
                .integerToOneHot("Num Of Products", 1, 4)
                .normalize("Tenure", Normalize.MinMax, analysis)
                .normalize("Age", Normalize.Standardize, analysis)
                .normalize("Credit Score", Normalize.Log2Mean, analysis)
                .normalize("Balance", Normalize.Log2MeanExcludingMin, analysis)
                .normalize("Estimated Salary", Normalize.Log2MeanExcludingMin, analysis)
                .build();

        finalSchema = transformProcess.getFinalSchema();
    }

    private void initSchema() {
        schema = new Schema.Builder()
                .addColumnsInteger("Row Number", "Customer Id")
                .addColumnString("Surname")
                .addColumnInteger("Credit Score")
                .addColumnCategorical("Geography", "France", "Germany", "Spain")
                .addColumnCategorical("Gender", "Female", "Male")
                .addColumnsInteger("Age", "Tenure")
                .addColumnDouble("Balance")
                .addColumnInteger("Num Of Products")
                .addColumnCategorical("Has Credit Card", "0", "1")
                .addColumnCategorical("Is Active Member", "0", "1")
                .addColumnDouble("Estimated Salary")
                .addColumnCategorical("Exited", "0", "1")
                .build();
    }

    private void analiseData() {
        try {
            analysis = AnalyzeLocal.analyze(schema, recordReader, 30);
            HtmlAnalysis.createHtmlAnalysisFile(analysis, new File(pathHtmlAnalise));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public String getModelPath() {
        return modelPath;
    }
}
