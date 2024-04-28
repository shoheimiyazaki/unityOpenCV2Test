namespace OpenCvSharp.Test
{
    using System.Collections;
    using System.Collections.Generic;
    using UnityEngine;
    using System;
    using System.Collections.Generic;
    using OpenCvSharp;
    using Rect = OpenCvSharp.Rect;
    using System.Linq;

    public class test : MonoBehaviour
    {
        public TextAsset faces;
        public TextAsset eyes;
        public TextAsset shapes;

        private WebCamTexture webCamTexture = null;

        protected CascadeClassifier cascadeFaces = null;
        protected CascadeClassifier cascadeEyes = null;
        protected Mat processingImage = null;
        protected Double appliedFactor = 1.0;
        protected Unity.TextureConversionParams TextureParameters { get; private set; }

        public Mat Image { get; private set; }
        public FaceProcessorPerformanceParams Performance { get; private set; }


        private void Awake()
        {
            if (webCamTexture == null)
            {
                var webCamIndex = WebCamTexture.devices.Length - 1;
                var device = WebCamTexture.devices[webCamIndex];
                var deviceName = device.name;
                webCamTexture = new WebCamTexture(deviceName);
                {
                    Unity.TextureConversionParams parameters = new Unity.TextureConversionParams();
                    parameters.FlipHorizontally = device.isFrontFacing;
                    if (0 != webCamTexture.videoRotationAngle)
                        parameters.RotationAngle = webCamTexture.videoRotationAngle; // cw -> ccw
                    TextureParameters = parameters;
                }
                webCamTexture.Play();
            }


            byte[] shapeDat = shapes.bytes;
            if (shapeDat.Length == 0)
            {
                string errorMessage =
                    "In order to have Face Landmarks working you must download special pre-trained shape predictor " +
                    "available for free via DLib library website and replace a placeholder file located at " +
                    "\"OpenCV+Unity/Assets/Resources/shape_predictor_68_face_landmarks.bytes\"\n\n" +
                    "Without shape predictor demo will only detect face rects.";

#if UNITY_EDITOR
                // query user to download the proper shape predictor
                if (UnityEditor.EditorUtility.DisplayDialog("Shape predictor data missing", errorMessage, "Download", "OK, process with face rects only"))
                    Application.OpenURL("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2");
#else
             UnityEngine.Debug.Log(errorMessage);
#endif
            }

            Initialize(faces.text, eyes.text, shapes.bytes);
            DataStabilizer = new DataStabilizerParams();
            Performance = new FaceProcessorPerformanceParams();

            // data stabilizer - affects face rects, face landmarks etc.
            DataStabilizer.Enabled = true;        // enable stabilizer
            DataStabilizer.Threshold = 2.0;       // threshold value in pixels
            DataStabilizer.SamplesCount = 2;      // how many samples do we need to compute stable data

            // performance data - some tricks to make it work faster
            Performance.Downscale = 256;          // processed image is pre-scaled down to N px by long side
            Performance.SkipRate = 0;             // we actually process only each Nth frame (and every frame for skipRate = 0)
        }
        public virtual void Initialize(string facesCascadeData, string eyesCascadeData, byte[] shapeData = null)
        {
            // face detector - the key thing here
            if (null == facesCascadeData || facesCascadeData.Length == 0)
                throw new Exception("FaceProcessor.Initialize: No face detector cascade passed, with parameter is required");

            FileStorage storageFaces = new FileStorage(facesCascadeData, FileStorage.Mode.Read | FileStorage.Mode.Memory);
            cascadeFaces = new CascadeClassifier();
            if (!cascadeFaces.Read(storageFaces.GetFirstTopLevelNode()))
                throw new System.Exception("FaceProcessor.Initialize: Failed to load faces cascade classifier");

            // eyes detector
            if (null != eyesCascadeData)
            {
                FileStorage storageEyes = new FileStorage(eyesCascadeData, FileStorage.Mode.Read | FileStorage.Mode.Memory);
                cascadeEyes = new CascadeClassifier();
                if (!cascadeEyes.Read(storageEyes.GetFirstTopLevelNode()))
                    throw new System.Exception("FaceProcessor.Initialize: Failed to load eyes cascade classifier");
            }

            // shape detector
            if (null != shapeData && shapeData.Length > 0)
            {
                shapeFaces = new ShapePredictor();
                shapeFaces.LoadData(shapeData);
            }
        }
        // Start is called before the first frame update
        void Start()
        {

        }
        protected virtual void ImportTexture(Texture texture, Unity.TextureConversionParams texParams)
        {
            // free currently used textures
            if (null != processingImage)
                processingImage.Dispose();
            if (null != Image)
                Image.Dispose();

            // convert and prepare
            Image = MatFromTexture(texture, texParams);
            if (Performance.Downscale > 0 && (Performance.Downscale < Image.Width || Performance.Downscale < Image.Height))
            {
                // compute aspect-respective scaling factor
                int w = Image.Width;
                int h = Image.Height;

                // scale by max side
                if (w >= h)
                {
                    appliedFactor = (double)Performance.Downscale / (double)w;
                    w = Performance.Downscale;
                    h = (int)(h * appliedFactor + 0.5);
                }
                else
                {
                    appliedFactor = (double)Performance.Downscale / (double)h;
                    h = Performance.Downscale;
                    w = (int)(w * appliedFactor + 0.5);
                }

                // resize
                processingImage = new Mat();
                Cv2.Resize(Image, processingImage, new Size(w, h));
            }
            else
            {
                appliedFactor = 1.0;
                processingImage = Image;
            }
        }
        protected virtual Mat MatFromTexture(Texture texture, Unity.TextureConversionParams texParams)
        {
            if (texture is UnityEngine.Texture2D)
                return Unity.TextureToMat(texture as UnityEngine.Texture2D, texParams);
            else if (texture is UnityEngine.WebCamTexture)
                return Unity.TextureToMat(texture as UnityEngine.WebCamTexture, texParams);
            else
                throw new Exception("FaceProcessor: incorrect input texture type, must be Texture2D or WebCamTexture");
        }
        // Update is called once per frame
        void Update()
        {
            if (webCamTexture == null)
            {
                var webCamIndex = WebCamTexture.devices.Length - 1;
                var device = WebCamTexture.devices[webCamIndex];
                var deviceName = device.name;
                webCamTexture = new WebCamTexture(deviceName);
                {
                    Unity.TextureConversionParams parameters = new Unity.TextureConversionParams();
                    parameters.FlipHorizontally = device.isFrontFacing;
                    if (0 != webCamTexture.videoRotationAngle)
                        parameters.RotationAngle = webCamTexture.videoRotationAngle; // cw -> ccw

                    // apply
                    TextureParameters = parameters;
                }
                webCamTexture.Play();
            }

            if (webCamTexture != null && webCamTexture.didUpdateThisFrame)
            {
                //ReadTextureConversionParameters();

                var faces = ProcessTexture(webCamTexture, TextureParameters);
                if (faces.Count > 0)
                {
                    var loc = faces[0].Region.Center;

                    var obj = GameObject.Find("Cube");
                    obj.transform.position = new Vector3(loc.X, loc.Y);

                }
            }
        }

        private DataStabilizerParams DataStabilizer { get; set; }
        private List<DetectedFace> Faces { get; set; }
        public bool cutFalsePositivesWithEyesSearch = true;
        protected ShapePredictor shapeFaces = null;

        private List<DetectedFace> ProcessTexture(Texture texture, Unity.TextureConversionParams texParams)
        {
            // convert Unity texture to OpenCv::Mat
            ImportTexture(texture, texParams);

            double invF = 1.0 / appliedFactor;
            DataStabilizer.ThresholdFactor = invF;

            // convert to grayscale and normalize
            Mat gray = new Mat();
            Cv2.CvtColor(processingImage, gray, ColorConversionCodes.BGR2GRAY);

            // fix shadows
            Cv2.EqualizeHist(gray, gray);

            /*Mat normalized = new Mat();
            CLAHE clahe = CLAHE.Create();
            clahe.TilesGridSize = new Size(8, 8);
            clahe.Apply(gray, normalized);
            gray = normalized;*/

            // detect matching regions (faces bounding)
            Rect[] rawFaces = cascadeFaces.DetectMultiScale(gray, 1.2, 6);
            if (Faces.Count != rawFaces.Length)
                Faces.Clear();

            // now per each detected face draw a marker and detect eyes inside the face rect
            int facesCount = 0;
            for (int i = 0; i < rawFaces.Length; ++i)
            {
                Rect faceRect = rawFaces[i];
                Rect faceRectScaled = faceRect * invF;
                using (Mat grayFace = new Mat(gray, faceRect))
                {
                    // another trick: confirm the face with eye detector, will cut some false positives
                    if (cutFalsePositivesWithEyesSearch && null != cascadeEyes)
                    {
                        Rect[] eyes = cascadeEyes.DetectMultiScale(grayFace);
                        if (eyes.Length == 0 || eyes.Length > 2)
                            continue;
                    }

                    // get face object
                    DetectedFace face = null;
                    if (Faces.Count < i + 1)
                    {
                        face = new DetectedFace(DataStabilizer, faceRectScaled);
                        Faces.Add(face);
                    }
                    else
                    {
                        face = Faces[i];
                        face.SetRegion(faceRectScaled);
                    }

                    // shape
                    facesCount++;
                    if (null != shapeFaces)
                    {
                        Point[] marks = shapeFaces.DetectLandmarks(gray, faceRect);

                        // we have 68-point predictor
                        if (marks.Length == 68)
                        {
                            // transform landmarks to the original image space
                            List<Point> converted = new List<Point>();
                            foreach (Point pt in marks)
                                converted.Add(pt * invF);

                            // save and parse landmarks
                            face.SetLandmarks(converted.ToArray());
                        }
                    }
                }
            }
            return Faces;
        }

    }
    public class FaceProcessorPerformanceParams
    {
        /// <summary>
        /// Downscale limit, texture processing will downscale input up to this size
        /// If is less or equals to zero than downscaling is not applied
        /// 
        /// Downscaling is applied with preserved aspect ratio
        /// </summary>
        public int Downscale { get; set; }

        /// <summary>
        /// Processor will skip that much frames before processing anything, 0 means no skip
        /// </summary>
        public int SkipRate { get; set; }

        /// <summary>
        /// Default constructor
        /// </summary>
        public FaceProcessorPerformanceParams()
        {
            Downscale = 0;
            SkipRate = 0;
        }
    }

    class DetectedObject
    {
        PointsDataStabilizer marksStabilizer = null;

        /// <summary>
        /// Default constructor
        /// </summary>
        /// <param name="stabilizerParameters">Data stabilizer params</param>
        public DetectedObject(DataStabilizerParams stabilizerParameters)
        {
            marksStabilizer = new PointsDataStabilizer(stabilizerParameters);
            marksStabilizer.PerPointProcessing = false;

            Marks = null;
            Elements = new DetectedObject[0];
        }

        /// <summary>
        /// Constructs object with name and region
        /// </summary>
        /// <param name="name">Detected objetc name</param>
        /// <param name="region">Detected object ROI on the source image</param>
        /// /// <param name="stabilizerParameters">Data stabilizer params</param>
        public DetectedObject(DataStabilizerParams stabilizerParameters, String name, Rect region)
            : this(stabilizerParameters)
        {
            Name = name;
            Region = region;
        }

        /// <summary>
        /// Constructs object with name and marks
        /// </summary>
        /// <param name="name">Detected object name</param>
        /// <param name="marks">Object landmarks (in the source image space)</param>
        /// /// <param name="stabilizerParameters">Data stabilizer params</param>
        public DetectedObject(DataStabilizerParams stabilizerParameters, String name, OpenCvSharp.Point[] marks)
            : this(stabilizerParameters)
        {
            Name = name;

            marksStabilizer.Sample = marks;
            Marks = marksStabilizer.Sample;

            Region = Rect.BoundingBoxForPoints(marks);
        }

        /// <summary>
        /// Object name
        /// </summary>
        public String Name { get; protected set; }

        /// <summary>
        /// Object region on the source image
        /// </summary>
        public Rect Region { get; protected set; }

        /// <summary>
        /// Object key points
        /// </summary>
        public OpenCvSharp.Point[] Marks { get; protected set; }

        /// <summary>
        /// Sub-objects
        /// </summary>
        public DetectedObject[] Elements { get; set; }

        /// <summary>
        /// Applies new marks
        /// </summary>
        /// <param name="marks">New points set</param>
        /// <param name="stabilizer">Signals whether we should apply stabilizer</param>
        /// <returns>True is new data applied, false if stabilizer rejected new data</returns>
        public virtual bool SetMarks(Point[] marks)
        {
            marksStabilizer.Sample = marks;

            Marks = marksStabilizer.Sample;
            return marksStabilizer.LastApplied;
        }
    }
    class DetectedFace : DetectedObject
    {
        /// <summary>
        /// Face elements
        /// </summary>
        public enum FaceElements
        {
            Jaw = 0,

            LeftEyebrow,
            RightEyebrow,

            NoseBridge,
            Nose,

            LeftEye,
            RightEye,

            OuterLip,
            InnerLip
        }

        /// <summary>
        /// Simple 2d integer triangle
        /// </summary>
        public struct Triangle
        {
            public Point i;
            public Point j;
            public Point k;

            /// <summary>
            /// Special constructor
            /// </summary>
            /// <param name="vec">Vec containing triangle (like those returned by Subdiv.gettrianglesList() method)</param>
            public Triangle(Vec6f vec)
            {
                i = new Point((int)(vec[0] + 0.5), (int)(vec[1] + 0.5));
                j = new Point((int)(vec[2] + 0.5), (int)(vec[3] + 0.5));
                k = new Point((int)(vec[4] + 0.5), (int)(vec[5] + 0.5));
            }

            /// <summary>
            /// Converts triangle to points array
            /// </summary>
            /// <returns>Array of triangle points</returns>
            public Point[] ToArray()
            {
                return new Point[] { i, j, k };
            }
        }

        /// <summary>
        /// Face data like convex hull, delaunay triangulation etc.
        /// </summary>
        public sealed class FaceInfo
        {
            /// <summary>
            /// Face shape convex hull
            /// </summary>
            public Point[] ConvexHull { get; private set; }

            /// <summary>
            /// Face shape triangulation
            /// </summary>
            public Triangle[] DelaunayTriangles { get; private set; }

            /// <summary>
            /// Constructs face info
            /// </summary>
            /// <param name="hull">Convex hull</param>
            /// <param name="triangles">Delaunay triangulation data</param>
            internal FaceInfo(Point[] hull, Triangle[] triangles)
            {
                ConvexHull = hull;
                DelaunayTriangles = triangles;
            }
        }

        /// <summary>
        /// Face info, heavy and lazy-computed data
        /// </summary>
        public FaceInfo Info
        {
            get
            {
                if (null == faceInfo)
                {
                    // it's valid to have no marks (no shape predictor used)
                    if (null == Marks)
                        return null;

                    // convex hull
                    Point[] hull = Cv2.ConvexHull(Marks);

                    // compute triangles
                    Rect bounds = Rect.BoundingBoxForPoints(hull);
                    Subdiv2D subdiv = new Subdiv2D(bounds);
                    foreach (Point pt in Marks)
                        subdiv.Insert(pt);

                    Vec6f[] vecs = subdiv.GetTriangleList();
                    List<Triangle> triangles = new List<Triangle>();
                    for (int i = 0; i < vecs.Length; ++i)
                    {
                        Triangle t = new Triangle(vecs[i]);
                        if (bounds.Contains(t.ToArray()))
                            triangles.Add(t);
                    }

                    // save
                    faceInfo = new FaceInfo(hull, triangles.ToArray());
                }
                return faceInfo;
            }
        }

        protected FaceInfo faceInfo = null;
        RectStabilizer faceStabilizer = null;

        /// <summary>
        /// Constructs DetectedFace object
        /// </summary>
        /// <param name="roi">Face roi (rectangle) in the source image space</param>
        /// /// <param name="stabilizerParameters">Data stabilizer params</param>
        public DetectedFace(DataStabilizerParams stabilizerParameters, Rect roi)
            : base(stabilizerParameters, "Face", roi)
        {
            faceStabilizer = new RectStabilizer(stabilizerParameters);
        }

        /// <summary>
        /// Sets face rect
        /// </summary>
        /// <param name="roi">Face rect</param>
        public void SetRegion(Rect roi)
        {
            faceStabilizer.Sample = roi;

            Region = faceStabilizer.Sample;
            faceInfo = null;
        }

        /// <summary>
        /// Creates new sub-object
        /// </summary>
        /// <param name="element">Face element type</param>
        /// <param name="name">New object name</param>
        /// <param name="fromMark">Starting mark index</param>
        /// <param name="toMark">Ending mark index</param>
        /// <param name="factor">Scale factor</param>
        /// <param name="updateMarks">[optional] Signals whether we should apply new marks</param>
        public bool DefineSubObject(FaceElements element, string name, int fromMark, int toMark, bool updateMarks = true)
        {
            int index = (int)element;
            Point[] subset = Marks.SubsetFromTo(fromMark, toMark);
            DetectedObject obj = Elements[index];

            // first instance
            bool applied = false;
            if (null == obj)
            {
                applied = true;
                obj = new DetectedObject(faceStabilizer.Params, name, subset);
                Elements[index] = obj;
            }
            // updated
            else
            {
                if (updateMarks || null == obj.Marks || 0 == obj.Marks.Length)
                    applied = obj.SetMarks(subset);
            }

            return applied;
        }

        /// <summary>
        /// Sets face landmarks
        /// </summary>
        /// <param name="points">New landmarks set</param>
        public void SetLandmarks(Point[] points)
        {
            // set marks
            Marks = points;

            // apply subs
            if (null == Elements || Elements.Length < 9)
                Elements = new DetectedObject[9];
            int keysApplied = 0;

            // key elements
            if (null != Marks)
            {
                keysApplied += DefineSubObject(FaceElements.Nose, "Nose", 30, 35) ? 1 : 0;
                keysApplied += DefineSubObject(FaceElements.LeftEye, "Eye", 36, 41) ? 1 : 0;
                keysApplied += DefineSubObject(FaceElements.RightEye, "Eye", 42, 47) ? 1 : 0;

                // non-key but independent
                DefineSubObject(FaceElements.OuterLip, "Lip", 48, 59);
                DefineSubObject(FaceElements.InnerLip, "Lip", 60, 67);

                // dependent
                bool updateDependants = keysApplied > 0;
                DefineSubObject(FaceElements.LeftEyebrow, "Eyebrow", 17, 21, updateDependants);
                DefineSubObject(FaceElements.RightEyebrow, "Eyebrow", 22, 26, updateDependants);
                DefineSubObject(FaceElements.NoseBridge, "Nose bridge", 27, 30, updateDependants);
                DefineSubObject(FaceElements.Jaw, "Jaw", 0, 16, updateDependants);
            }

            // re-fetch marks from sub-objects as they have separate stabilizers
            List<Point> fetched = new List<Point>();
            foreach (DetectedObject obj in Elements)
                if (obj.Marks != null)
                    fetched.AddRange(obj.Marks);
            Marks = fetched.ToArray();

            // drop cache
            faceInfo = null;
        }
    }
    class DataStabilizerParams
    {
        /// <summary>
        /// Should this stabilizer just push data through (false value) or do some work before (true value)?
        /// </summary>
        public bool Enabled { get; set; }

        /// <summary>
        /// Maximum ignored point distance
        /// </summary>
        public double Threshold { get; set; }

        /// <summary>
        /// Threshold scale factor (should processing space be scaled)
        /// </summary>
        public double ThresholdFactor { get; set; }

        /// <summary>
        /// Accumulated samples count
        /// </summary>
        public int SamplesCount { get; set; }

        /// <summary>
        /// Returns scaled threshold
        /// </summary>
        /// <returns></returns>
        public double GetScaledThreshold()
        {
            return Threshold * ThresholdFactor;
        }

        /// <summary>
        /// Default constructor
        /// </summary>
        public DataStabilizerParams()
        {
            Enabled = true;
            Threshold = 1.0;
            ThresholdFactor = 1.0;
            SamplesCount = 10;
        }
    }


    /// <summary>
    /// On top of various OpenCV stabilizers for video itself (like optical flow) we might
    /// need a simpler one "stabilizer" for some data reacquired each frame like face rects,
    /// face landmarks etc.
    /// 
    /// This class is designed to be fast, so it basically applies some threshold and simplest heuristics
    /// to decide whether to update it's data set with new data chunk
    /// </summary>
    class PointsDataStabilizer : DataStabilizerBase<Point[]>
    {
        /// <summary>
        /// Flag signaling whether data set is interpreted as whole (Triangle, Rectangle) or independent (independent points array)
        /// </summary>
        public bool PerPointProcessing { get; set; }

        /// <summary>
        /// Creates DataStabilizer instance
        /// </summary>
        /// <param name="parameters">Stabilizer general parameters</param>
        public PointsDataStabilizer(DataStabilizerParams parameters)
            : base(parameters)
        {
            PerPointProcessing = true;
        }

        /// <summary>
        /// Validate sample
        /// </summary>
        /// <param name="sample"></param>
        protected override void ValidateSample(Point[] sample)
        {
            if (null == sample || sample.Length == 0)
                throw new ArgumentException("sample: is null or empty array.");

            foreach (Point[] data in samples)
            {
                if (data != null && data.Length != sample.Length)
                    throw new ArgumentException("sample: invalid input data, length does not match.");
            }
        }

        /// <summary>
        /// Computes average data sample
        /// </summary>
        /// <returns></returns>
        protected override Point[] ComputeAverageSample()
        {
            // we need full stack to run
            if (inputSamples < Params.SamplesCount)
                return null;

            // accumulate average
            int sampleSize = samples[0].Length;
            Point[] average = new Point[sampleSize];
            for (int s = 0; s < Params.SamplesCount; ++s)
            {
                Point[] data = samples[s];
                for (int i = 0; i < sampleSize; ++i)
                    average[i] += data[i];
            }

            // normalize
            double inv = 1.0 / Params.SamplesCount;
            for (int i = 0; i < sampleSize; ++i)
                average[i] = new Point(average[i].X * inv + 0.5, average[i].Y * inv);
            return average;
        }

        /// <summary>
        /// Computes data sample
        /// </summary>
        /// <returns>True if final sample changed, false if current frame is the same as the last one</returns>
        protected override bool PrepareStabilizedSample()
        {
            // get average
            Point[] average = ComputeAverageSample();
            if (null == average)
                return false;

            // if we have no saved result at all - average will do
            if (DefaultValue() == result)
            {
                result = average;
                return true;
            }

            // we have new average and saved data as well - test it
            double dmin = double.MaxValue, dmax = double.MinValue, dmean = 0.0;
            double[] distance = new double[result.Length];
            for (int i = 0; i < result.Length; ++i)
            {
                double d = Point.Distance(result[i], average[i]);

                dmean += d;
                dmax = Math.Max(dmax, d);
                dmin = Math.Min(dmin, d);
                distance[i] = d;
            }
            dmean /= result.Length;

            // check whether it's OK to apply
            double edge = Params.Threshold;
            if (dmean > edge)
            {
                result = average;
                return true;
            }

            // per-item process
            bool anyChanges = false;
            if (PerPointProcessing)
            {
                for (int i = 0; i < result.Length; ++i)
                {
                    if (distance[i] > edge)
                    {
                        anyChanges = true;
                        result[i] = average[i];
                    }
                }
            }
            return anyChanges;
        }

        /// <summary>
        /// Gets default value for the output sample
        /// </summary>
        /// <returns></returns>
        protected override Point[] DefaultValue()
        {
            return null;
        }
    }

    abstract class DataStabilizerBase<T>
    {
        protected T result;                     // computer output sample
        protected bool dirty = true;            // flag signals whether "result" sample must be recomputed
        protected T[] samples = null;           // whole samples set
        protected long inputSamples = 0;        // processed samples count

        /// <summary>
        /// Parameters, see corresponding class
        /// </summary>
        public DataStabilizerParams Params { get; set; }

        /// <summary>
        /// Stabilized data
        /// </summary>
        public virtual T Sample
        {
            get
            {
                // requires update
                if (dirty)
                {
                    // samples count changed
                    if (samples.Length != Params.SamplesCount)
                    {
                        T[] data = new T[Params.SamplesCount];
                        Array.Copy(samples, data, Math.Min(samples.Length, Params.SamplesCount));
                        samples = data;

                        // drop result
                        result = DefaultValue();
                    }

                    // prepare to compute
                    LastApplied = true;

                    // process samples
                    if (Params.Enabled)
                        LastApplied = PrepareStabilizedSample();
                    // stabilizer is disabled - simply grab the fresh-most sample
                    else
                        result = samples[0];
                    dirty = false;
                }
                return result;
            }
            set
            {
                ValidateSample(value);

                // shift and push new value to the top
                T[] data = new T[Params.SamplesCount];
                Array.Copy(samples, 0, data, 1, Params.SamplesCount - 1);
                data[0] = value;
                samples = data;
                inputSamples++;

                // mark
                dirty = true;
            }
        }

        /// <summary>
        /// Signals whether last data chunk changed anything
        /// </summary>
        public bool LastApplied { get; private set; }

        /// <summary>
        /// Constructs base data stabilizer
        /// </summary>
        protected DataStabilizerBase(DataStabilizerParams parameters)
        {
            Params = parameters;
            samples = new T[Params.SamplesCount];
            result = DefaultValue();
        }

        /// <summary>
        /// Computes stabilized data sample
        /// </summary>
        /// <returns>True if data has been recomputed, false if nothing changed for returned sample</returns>
        protected abstract bool PrepareStabilizedSample();

        /// <summary>
        /// Computes average data sample
        /// </summary>
        /// <returns></returns>
        protected abstract T ComputeAverageSample();

        /// <summary>
        /// Tests sample validity, must throw on unexpected value
        /// </summary>
        /// <param name="sample">Sample to test</param>
        protected abstract void ValidateSample(T sample);

        /// <summary>
        /// Gets default value for the output sample
        /// </summary>
        /// <returns></returns>
        protected abstract T DefaultValue();
    }

    static partial class ArrayUtilities
    {
        // create a subset from a range of indices
        public static T[] RangeSubset<T>(this T[] array, int startIndex, int length)
        {
            T[] subset = new T[length];
            Array.Copy(array, startIndex, subset, 0, length);
            return subset;
        }

        // creates subset with from-to index pair
        public static T[] SubsetFromTo<T>(this T[] array, int fromIndex, int toIndex)
        {
            return array.RangeSubset<T>(fromIndex, toIndex - fromIndex + 1);
        }

        // create a subset from a specific list of indices
        public static T[] Subset<T>(this T[] array, params int[] indices)
        {
            T[] subset = new T[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                subset[i] = array[indices[i]];
            }
            return subset;
        }
    }

    /// <summary>
    /// Data stabilizer designed for OpenCv Rect (Object tracking, face detection etc.)
    /// </summary>
    class RectStabilizer : DataStabilizerBase<Rect>
    {
        /// <summary>
        /// Constructs Rectangle stabilizer
        /// </summary>
        /// <param name="parameters">Data stabilizer general parameters</param>
        public RectStabilizer(DataStabilizerParams parameters)
            : base(parameters)
        { }

        /// <summary>
        /// Computes average data sample
        /// </summary>
        /// <returns></returns>
        protected override Rect ComputeAverageSample()
        {
            Rect average = new Rect();
            if (inputSamples < Params.SamplesCount)
                return average;

            foreach (Rect rc in samples)
                average = average + rc;
            return average * (1.0 / Params.SamplesCount);
        }

        /// <summary>
        /// For Rect stabilizer any sample is valid
        /// </summary>
        /// <param name="sample">Sample to test</param>
        protected override void ValidateSample(Rect sample)
        { }

        /// <summary>
        /// Prepares stabilized sample (Rectangle)
        /// </summary>
        protected override bool PrepareStabilizedSample()
        {
            Rect average = ComputeAverageSample();

            // quick check
            if (DefaultValue() == result)
            {
                result = average;
                return true;
            }

            // compute per-corner distance between the frame we have and new one
            double dmin = double.MaxValue, dmax = double.MinValue, dmean = 0.0;
            Point[] our = result.ToArray(), their = average.ToArray();
            for (int i = 0; i < 4; ++i)
            {
                double distance = Point.Distance(our[i], their[i]);
                dmin = Math.Min(distance, dmin);
                dmax = Math.Max(distance, dmax);
                dmean += distance;
            }
            dmean /= their.Length;

            // apply conditions
            if (dmin > Params.GetScaledThreshold())
            {
                result = average;
                return true;
            }

            return false;
        }

        /// <summary>
        /// Gets default value for the output sample
        /// </summary>
        /// <returns></returns>
        protected override Rect DefaultValue()
        {
            return new Rect();
        }
    }
}