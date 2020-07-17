/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package finalgraphweighted;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.List;
import java.util.StringTokenizer;
import java.util.Scanner;
import org.jblas.DoubleMatrix;
import static org.jblas.Geometry.normalizeColumns;
import org.jgraph.graph.DefaultEdge;
import org.jgrapht.UndirectedGraph;
import org.jgrapht.alg.interfaces.VertexScoringAlgorithm;
import org.jgrapht.alg.scoring.PageRank;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.DirectedPseudograph;
import org.jgrapht.graph.DirectedWeightedPseudograph;
import org.jgrapht.graph.SimpleGraph;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Iterator;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

/**
 *
 * @author farzane
 */
public class Finalgraphweighted {

    public static DoubleMatrix signal_similarity(int nodeCount, int t, DoubleMatrix adjMatrix) {

        DoubleMatrix I = DoubleMatrix.eye(nodeCount);

        DoubleMatrix signalMatrix = new DoubleMatrix(nodeCount, nodeCount);
        signalMatrix = (adjMatrix.add(I));

        for (int i = 0; i < t - 1; i++) {
            signalMatrix = signalMatrix.mmul(adjMatrix.add(I));
        }

        double s = 0;
        int i = 0;
        int j1 = 0, j2 = 0;
        for (i = 0; i < nodeCount; i++) {
            s = 0;
            for (j1 = 0; j1 < nodeCount; j1++) {
                s = s + Math.pow(signalMatrix.get(j1, i), 2);
            }
            s = Math.sqrt(s);
            for (j2 = 0; j2 < nodeCount; j2++) {
                signalMatrix.put(j2, i, signalMatrix.get(j2, i) / s);
            }
        }
        return signalMatrix;
    }

    public static double[] pageRank(DoubleMatrix m) {

        DirectedPseudograph<String, DefaultWeightedEdge> g
                = new DirectedWeightedPseudograph<String, DefaultWeightedEdge>(DefaultWeightedEdge.class);

        double[] page_rank = new double[m.rows];
        int k = 0;
        for (int i = 0; i < m.rows; i++) {
            g.addVertex("" + i);
        }
        int j = 0;
        for (int i = 0; i < m.rows; i++) {
            for (j = 0; j < m.rows; j++) {
                if (i != j) {

                    g.setEdgeWeight(g.addEdge("" + i, "" + j), 1 / (0.00001 + m.get(i, j)));
                }
            }
        }
        VertexScoringAlgorithm<String, Double> pr = new PageRank(g, 0.85, 1000, 0.0001);

        // pageRank.compute();
        for (int i = 0; i < m.rows; i++) {

            page_rank[i] = pr.getVertexScore("" + i);
        }
        return page_rank;
    }

    public static double distance(DoubleMatrix signalMatrix, int m, int n) {
        double c = 0;
        for (int j = 0; j < signalMatrix.rows; j++) {
            c += Math.pow(signalMatrix.get(j, m) - signalMatrix.get(j, n), 2);
        }
        return Math.sqrt(c);
    }

    public static ArrayList knearestneighbor(DoubleMatrix signalMatrix, int m, int nodeCount, int k) {
        Hashtable distanceNode = new Hashtable();
        double[] keys = new double[nodeCount];
        ArrayList neighbor = new ArrayList();
        double tmp = 0;
        for (int i = 0; i < nodeCount; i++) {
            if (i != m) {
                for (int j = 0; j < signalMatrix.rows; j++) {
                    tmp += Math.pow(signalMatrix.get(j, m) - signalMatrix.get(j, i), 2);
                }
                distanceNode.put(Math.sqrt(tmp), i);
                keys[i] = Math.sqrt(tmp);
            }
            tmp = 0;
        }

        Arrays.sort(keys);

        for (int l = 1; l <= k; l++) {
            neighbor.add(distanceNode.get(keys[l]));
        }

        return neighbor;
    }

    public static void writeMatrix(String filename, DoubleMatrix matrix, String dataset) {

        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
            bw.write("% The " + dataset + " data");
            bw.newLine();
            bw.newLine();
            bw.write("@relation " + dataset);
            bw.newLine();
            bw.newLine();
            for (int i = 0; i < matrix.rows; i++) {
                bw.write("@attribute node" + i + " numeric");
                bw.newLine();
            }
            bw.newLine();
            bw.write("@data");
            bw.newLine();

            for (int i = 0; i < matrix.rows; i++) {
                for (int j = 0; j < matrix.rows; j++) {
                    if (j == matrix.rows - 1) {
                        bw.write(String.valueOf(matrix.get(i, j)));
                    } else {
                        bw.write(String.valueOf(matrix.get(i, j)) + ",");
                    }
                }
                bw.newLine();
            }
            bw.flush();
        } catch (IOException e) {
            // Prints what exception has been thrown 
            System.out.println(e);
        }
    }

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    public static void Kmeans(ArrayList seed, int clusters, String dataset) throws Exception {
        SimpleKMeans kmeans = new SimpleKMeans();

        Iterator<Integer> itr = seed.iterator();
        while (itr.hasNext()) {
            kmeans.setSeed(itr.next());
        }

        kmeans.setPreserveInstancesOrder(true);
        kmeans.setNumClusters(clusters);

        BufferedReader datafile = readDataFile(dataset + ".arff");
        Instances data = new Instances(datafile);

        kmeans.buildClusterer(data);

        // This array returns the cluster number (starting with 0) for each instance
        // The array has as many elements as the number of instances
        int[] assignments = kmeans.getAssignments();

        int i = 0;
        for (int clusterNum : assignments) {
            System.out.printf("Instance %d -> Cluster %d \n", i, clusterNum);
            i++;
        }
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {

        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter dataset name: ");
        String dataset = scanner.next();
        // TODO code application logic here
        DoubleMatrix adjMatrix, signalMatrix;
        List<Integer> list = new ArrayList<Integer>();
        File file = new File("test.gml");
        BufferedReader reader = null;
        double matrix[][] = null;
        int nodeCount = 0;

        try {
            reader = new BufferedReader(new FileReader(file));
            String text = null;
            int startId = 0;

            StringTokenizer st;
            int s = 0, t = 0, directedFlag = 2, w = 0;
            ArrayList<Integer> arr = new ArrayList<>();
            while ((text = reader.readLine()) != null) {
                st = new StringTokenizer(text);
                while (st.hasMoreTokens()) {
                    String currentToken = st.nextToken();

                    if (currentToken.equals("node")) {
                        nodeCount++;

                    }
                    if (currentToken.equals("id")) {
                        if ((Integer.parseInt(st.nextToken())) == 0) {
                            startId = 1;
                        }

                    }

                    if (currentToken.equals("directed")) {
                        directedFlag = Integer.parseInt(st.nextToken());
                    }

                    if (currentToken.equals("source")) {
                        s = Integer.parseInt(st.nextToken());
                    }
                    if (currentToken.equals("target")) {
                        t = Integer.parseInt(st.nextToken());

                    }
                    if (currentToken.equals("value")) {
                        w = Integer.parseInt(st.nextToken());

                    }

                    if (currentToken.equals("edge")) {
                        if (s == 0 && t == 0 && w == 0) {
                        } else {
                            arr.add(s);
                            arr.add(t);
                            arr.add(w);
                        }

                    }

                }

            }

            if (startId == 0) {

                nodeCount++;
            }

            adjMatrix = new DoubleMatrix(nodeCount, nodeCount);
            signalMatrix = new DoubleMatrix(nodeCount, nodeCount);
            int row = 0, col = 0, weight = 0;
            for (int i = 0; i < arr.size(); i += 3) {

                row = arr.get(i);
                col = arr.get(i + 1);
                weight = arr.get(i + 2);

                adjMatrix.put(row, col, 1*weight);
                if (directedFlag == 0) {

                    adjMatrix.put(col, row, 1 * weight);

                }
            }

            signalMatrix = signal_similarity(nodeCount, 3, adjMatrix);
            
            signalMatrix.transpose();
           
            writeMatrix(dataset + ".arff", signalMatrix, dataset);
            double[] page_ranks = new double[nodeCount];
            page_ranks = pageRank(signalMatrix);

            DoubleMatrix disMatrix = new DoubleMatrix(nodeCount, nodeCount);
            double[] distance_result = new double[nodeCount];

            Hashtable PagerankNode = new Hashtable();
            Hashtable NodePagerank = new Hashtable();
            Hashtable ScoreNode = new Hashtable();

            for (int c = 0; c < nodeCount; c++) {
                PagerankNode.put(page_ranks[c], c);
                NodePagerank.put(c, page_ranks[c]);

            }

            Arrays.sort(page_ranks);
            double tempe;

            ArrayList selected = new ArrayList();

            ArrayList neighbor = new ArrayList();
            int a = page_ranks.length;
            a--;

            double[] distance_temp = new double[nodeCount];
            double[] score_temp = new double[nodeCount];
            selected.add(PagerankNode.get(page_ranks[a]));
            int selectedItem = (int) selected.get(0);
            double tempdist = 0, minDist = 10000000000000.0;
            int oldsize, index, newsize = 0;
            do {
                NodePagerank.put(selectedItem, (double) 0);
                    
                neighbor.addAll(knearestneighbor(signalMatrix, selectedItem, nodeCount,2));

                oldsize = selected.size();
                for (int i = 0; i < nodeCount; i++) {

                    for (int l = 0; l < oldsize; l++) {
                        int tempSelectedItem = (int) selected.get(l);

                        tempdist = distance(signalMatrix, tempSelectedItem, i);
                        if (tempdist < minDist) {
                            minDist = tempdist;
                        }
                    }
                    score_temp[i] = minDist * (double) NodePagerank.get(i);
                    ScoreNode.put(score_temp[i], i);
                    minDist = 10000000000000.0;

                }
                Arrays.sort(score_temp);

                int maxscorecount = nodeCount;

                selectedItem = (int) ScoreNode.get(score_temp[--maxscorecount]);
                while (neighbor.contains(selectedItem) && maxscorecount > 0) {
                    maxscorecount--;
                    selectedItem = (int) ScoreNode.get(score_temp[maxscorecount]);
                }
                if (!selected.contains(selectedItem)) {

                    selected.add(selectedItem);
                }
                newsize = selected.size();
            } while (newsize > oldsize);

            System.out.println("seeds are :" + selected.toString());
            System.out.println("# of clusters are :" + selected.size());
            
            Kmeans(selected, newsize, dataset);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (reader != null) {
                    reader.close();
                }
            } catch (IOException e) {
            }

        }

    }

}
