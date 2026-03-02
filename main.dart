import 'dart:async';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:tflite/tflite.dart';
import 'dart:typed_data';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();

  runApp(AgroVisionApp(cameras));
}

class AgroVisionApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  AgroVisionApp(this.cameras);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: HomeScreen(cameras: cameras),
    );
  }
}

class HomeScreen extends StatelessWidget {
  final List<CameraDescription> cameras;

  HomeScreen({required this.cameras});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'AgroVision',
          style: TextStyle(
            fontSize: 24.0,
            fontWeight: FontWeight.bold,
          ),
        ),
        centerTitle: true,
        backgroundColor: Colors.green,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            HomeFeatureButton(
              icon: Icons.camera,
              title: 'Soil Detection',
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => SoilDetectionScreen(),
                  ),
                );
              },
            ),
            HomeFeatureButton(
              icon: Icons.agriculture,
              title: 'Crop Recommendation',
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => CropRecommendationScreen(),
                  ),
                );
              },
            ),
            HomeFeatureButton(
              icon: Icons.assessment,
              title: 'Yield Prediction',
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => YieldPredictionScreen(),
                  ),
                );
              },
            ),
            HomeFeatureButton(
              icon: Icons.grass,
              title: 'Weed Detection',
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) =>
                        WeedDetectionScreen(camera: cameras[0]),
                  ),
                );
              },
            ),
            HomeFeatureButton(
              icon: Icons.bug_report,
              title: 'Disease Detection',
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => DiseaseDetectionScreen(),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}

class HomeFeatureButton extends StatelessWidget {
  final IconData icon;
  final String title;
  final Function() onPressed;

  HomeFeatureButton(
      {required this.icon, required this.title, required this.onPressed});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 200.0,
      margin: EdgeInsets.only(bottom: 16.0),
      child: GestureDetector(
        onTap: onPressed,
        child: FeatureCard(icon: icon, title: title),
      ),
    );
  }
}

class FeatureCard extends StatelessWidget {
  final IconData icon;
  final String title;

  FeatureCard({required this.icon, required this.title});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 200.0,
      padding: EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(10.0),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.5),
            spreadRadius: 2,
            blurRadius: 5,
            offset: Offset(0, 3),
          ),
        ],
      ),
      child: Column(
        children: [
          Icon(
            icon,
            size: 48.0,
            color: Colors.green,
          ),
          SizedBox(height: 16.0),
          Text(
            title,
            style: TextStyle(
              fontSize: 18.0,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}

class SoilDetectionScreen extends StatefulWidget {
  @override
  _SoilDetectionScreenState createState() => _SoilDetectionScreenState();
}

class _SoilDetectionScreenState extends State<SoilDetectionScreen> {
  CameraController? _cameraController;
  bool _isModelReady = false;
  String _detectionResult = 'Waiting for detection...';
  double _accuracy = 0.0;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadModel();
  }

  // Initialize the camera
  void _initializeCamera() async {
    final cameras = await availableCameras();
    _cameraController = CameraController(cameras[0], ResolutionPreset.medium);
    await _cameraController!.initialize();
    if (mounted) {
      setState(() {
        _isModelReady = true;
        _cameraController!.startImageStream(_detectSoilCondition);
      });
    }
  }

  // Load the TensorFlow Lite model
  void _loadModel() async {
    try {
      await Tflite.loadModel(
        model: 'assets/soil.tflite',
        labels: 'assets/soillabels.txt',
      );
    } catch (e) {
      print('Error loading the model: $e');
    }
  }

  // Detect soil condition from the camera feed
  void _detectSoilCondition(CameraImage image) async {
    if (!_isModelReady) {
      return;
    }

    try {
      List<dynamic>? output = await Tflite.runModelOnFrame(
        bytesList: image.planes.map((plane) => plane.bytes).toList(),
        imageHeight: image.height,
        imageWidth: image.width,
        imageMean: 127.5,
        imageStd: 127.5,
        numResults: 2,
      );

      if (output != null && output.isNotEmpty) {
        setState(() {
          _detectionResult = output[0]['label'];
          _accuracy = output[0]['confidence'] * 100;
        });
      }
    } catch (e) {
      print('Error detecting soil condition: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Soil Detection'),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          if (_cameraController != null &&
              _cameraController!.value.isInitialized)
            AspectRatio(
              aspectRatio: 1,
              child: CameraPreview(_cameraController!),
            ),
          SizedBox(height: 16.0),
          Text('Soil Condition: $_detectionResult'),
          Text('Accuracy: ${_accuracy.toStringAsFixed(2)}%'),
        ],
      ),
    );
  }

  @override
  void dispose() {
    Tflite.close();
    _cameraController?.dispose();
    super.dispose();
  }
}

class CropRecommendationScreen extends StatefulWidget {
  @override
  _CropRecommendationScreenState createState() =>
      _CropRecommendationScreenState();
}

class _CropRecommendationScreenState extends State<CropRecommendationScreen> {
  final _formKey = GlobalKey<FormState>();
  double? _nitrogen;
  double? _phosphorus;
  double? _potassium;
  double? _temperature;
  double? _humidity;
  double? _ph;
  double? _rainfall;
  String? _detectedCrop;
  List<String> _labels = [];

  @override
  void initState() {
    super.initState();
    loadModelAndLabels();
  }

  Future<void> loadModelAndLabels() async {
    // Load TFLite model
    await Tflite.loadModel(
      model: 'assets/crop_recommendation_model.tflite',
      labels: 'assets/crop_labels.txt',
    );

    // Load label text file
    final labelsData = await rootBundle.loadString('assets/crop_labels.txt');
    _labels = labelsData.split('\n');
  }

  void _detectCrop() async {
    if (_formKey.currentState!.validate()) {
      _formKey.currentState!.save();

      // Prepare user input data as a list of double values
      final userInputs = [
        _nitrogen,
        _phosphorus,
        _potassium,
        _temperature,
        _humidity,
        _ph,
        _rainfall,
      ];

      final filteredUserInputs = userInputs
          .where((value) => value != null)
          .map((value) => value!)
          .toList();

      // Convert user input data to bytes
      final inputBytes =
          Float32List.fromList(filteredUserInputs).buffer.asUint8List();

      try {
        // Run the model on the input data
        List<dynamic>? output = await Tflite.runModelOnBinary(
          binary: inputBytes,
          numResults: 1, // Assuming you want only one result
        );

        if (output != null && output.isNotEmpty) {
          final recommendedCropIndex = output[0]['index'];
          setState(() {
            _detectedCrop = _labels[recommendedCropIndex];
          });
        }
      } catch (e) {
        print('Error detecting crop: $e');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Crop Recommendation'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              TextFormField(
                decoration: InputDecoration(labelText: 'Nitrogen (e.g., 66.5)'),
                keyboardType: TextInputType.number,
                onSaved: (value) {
                  _nitrogen = double.tryParse(value!);
                },
              ),
              TextFormField(
                decoration:
                    InputDecoration(labelText: 'Phosphorus (e.g., 45.2)'),
                keyboardType: TextInputType.number,
                onSaved: (value) {
                  _phosphorus = double.tryParse(value!);
                },
              ),
              TextFormField(
                decoration:
                    InputDecoration(labelText: 'Potassium (e.g., 32.0)'),
                keyboardType: TextInputType.number,
                onSaved: (value) {
                  _potassium = double.tryParse(value!);
                },
              ),
              TextFormField(
                decoration: InputDecoration(labelText: 'Temperature (°C)'),
                keyboardType: TextInputType.number,
                onSaved: (value) {
                  _temperature = double.tryParse(value!);
                },
              ),
              TextFormField(
                decoration: InputDecoration(labelText: 'Humidity (%)'),
                keyboardType: TextInputType.number,
                onSaved: (value) {
                  _humidity = double.tryParse(value!);
                },
              ),
              TextFormField(
                decoration: InputDecoration(labelText: 'pH (e.g., 6.5)'),
                keyboardType: TextInputType.number,
                onSaved: (value) {
                  _ph = double.tryParse(value!);
                },
              ),
              TextFormField(
                decoration: InputDecoration(labelText: 'Rainfall (mm)'),
                keyboardType: TextInputType.number,
                onSaved: (value) {
                  _rainfall = double.tryParse(value!);
                },
              ),
              ElevatedButton(
                onPressed: _detectCrop,
                child: Text('Submit'),
              ),
              if (_detectedCrop != null)
                Padding(
                  padding: EdgeInsets.all(16.0),
                  child: Text(
                    'Recommended Crop: $_detectedCrop',
                    style: TextStyle(fontSize: 20),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }
}

class YieldPredictionScreen extends StatefulWidget {
  @override
  _YieldPredictionScreenState createState() => _YieldPredictionScreenState();
}

class _YieldPredictionScreenState extends State<YieldPredictionScreen> {
  String? _selectedCrop;
  double? _landArea;
  double? _estimatedYield;

  final List<String> _crops = [
    "rice",
    "maize",
    "chickpea",
    "kidneybeans",
    "pigeonpeas",
    "mothbeans",
    "mungbean",
    "blackgram",
    "lentil",
    "pomegranate",
    "banana",
    "mango",
    "grapes",
    "watermelon",
    "muskmelon",
    "apple",
    "orange",
    "papaya",
    "coconut",
    "cotton",
    "jute",
    "coffee"
  ];

  void _predictYield() {
    if (_selectedCrop != null && _landArea != null) {
      double yieldFactor = 0.00;

      Map<String, double> yieldFactors = {
        "rice": 3.2,
        "maize": 9.4,
        "chickpea": 1.2,
        "kidneybeans": 1.06,
        "pigeonpeas": 1.8,
        "mothbeans": 1.21,
        "mungbean": 1.16,
        "blackgram": 1.13,
        "lentil": 1.3,
        "pomegranate": 10.0,
        "banana": 17.98,
        "mango": 19.0,
        "grapes": 20.0,
        "watermelon": 20.8,
        "muskmelon": 11.8,
        "apple": 10.0,
        "orange": 2.7,
        "papaya": 35.0,
        "coconut": 24.0,
        "cotton": 3.24,
        "jute": 10.66,
        "coffee": 0.55,
      };

      if (yieldFactors.containsKey(_selectedCrop)) {
        yieldFactor = yieldFactors[_selectedCrop]!;
        setState(() {
          _estimatedYield = _landArea! * yieldFactor;
        });
      } else {
        setState(() {
          _estimatedYield = null; // Crop not found, yield cannot be predicted
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Yield Prediction'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text('Yield Prediction'),
            DropdownButton<String>(
              value: _selectedCrop,
              items: _crops.map((crop) {
                return DropdownMenuItem<String>(
                  value: crop,
                  child: Text(crop),
                );
              }).toList(),
              onChanged: (String? newValue) {
                setState(() {
                  _selectedCrop = newValue;
                });
              },
              hint: Text('Select Crop'),
            ),
            TextFormField(
              decoration: InputDecoration(labelText: 'Land Area (Hectare)'),
              keyboardType: TextInputType.number,
              onChanged: (value) {
                _landArea = double.tryParse(value);
              },
            ),
            ElevatedButton(
              onPressed: _predictYield,
              child: Text('Predict Yield'),
            ),
            if (_estimatedYield != null)
              Padding(
                padding: EdgeInsets.all(16.0),
                child: Text(
                  'Estimated Yield: ${_estimatedYield?.toStringAsFixed(2)}  Metric Ton',
                  style: TextStyle(fontSize: 20),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class WeedDetectionScreen extends StatefulWidget {
  final CameraDescription camera;

  WeedDetectionScreen({required this.camera});

  @override
  _WeedDetectionScreenState createState() => _WeedDetectionScreenState();
}

class _WeedDetectionScreenState extends State<WeedDetectionScreen> {
  late CameraController _cameraController;
  List<Rect> _weedBoundingBoxes = [];
  bool _isModelReady = false;

  @override
  void initState() {
    super.initState();
    initializeCamera();
    loadModel();
  }

  void initializeCamera() async {
    _cameraController =
        CameraController(widget.camera, ResolutionPreset.medium);
    await _cameraController.initialize();
    if (!mounted) {
      return;
    }
    setState(() {
      _isModelReady = true;
      _cameraController.startImageStream((CameraImage image) {
        if (_isModelReady) {
          _detectWeed(image);
        }
      });
    });
  }

  void loadModel() async {
    try {
      await Tflite.loadModel(
        model: 'assets/weed.tflite',
        labels: 'assets/weedlabels.txt',
      );
    } catch (e) {
      print('Error loading the model: $e');
    }
  }

  void _detectWeed(CameraImage image) async {
    try {
      List<dynamic>? output = await Tflite.runModelOnFrame(
        bytesList: image.planes.map((plane) => plane.bytes).toList(),
        imageHeight: image.height,
        imageWidth: image.width,
        imageMean: 127.5,
        imageStd: 127.5,
        numResults: 2,
      );

      if (output != null && output.isNotEmpty) {
        setState(() {
          // Clear any existing boxes
          _weedBoundingBoxes.clear();

          // Get the detected boxes from the model output
          for (int i = 0; i < output.length; i++) {
            final Map<String, dynamic> box = output[i];
            final double top = box['rect']['y'];
            final double left = box['rect']['x'];
            final double bottom = box['rect']['y'] + box['rect']['h'];
            final double right = box['rect']['x'] + box['rect']['w'];
            _weedBoundingBoxes
                .add(Rect.fromPoints(Offset(left, top), Offset(right, bottom)));
          }
        });
      }
    } catch (e) {
      print('Error detecting weed: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_cameraController.value.isInitialized) {
      return Container();
    }

    return Stack(
      children: [
        CameraPreview(_cameraController),
        for (Rect boundingBox in _weedBoundingBoxes)
          CustomPaint(
            painter: WeedBoundingBoxPainter(boundingBox),
          ),
      ],
    );
  }

  @override
  void dispose() {
    Tflite.close();
    _cameraController.dispose();
    super.dispose();
  }
}

class WeedBoundingBoxPainter extends CustomPainter {
  final Rect boundingBox;

  WeedBoundingBoxPainter(this.boundingBox);

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    canvas.drawRect(boundingBox, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return false;
  }
}

class DiseaseDetectionScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Disease Detection'),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          ElevatedButton(
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                    builder: (context) => CornDiseaseDetectionScreen()),
              );
            },
            child: Text('Corn Disease Detection'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                    builder: (context) => PotatoDiseaseDetectionScreen()),
              );
            },
            child: Text('Potato Disease Detection'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                    builder: (context) => RiceDiseaseDetectionScreen()),
              );
            },
            child: Text('Rice Disease Detection'),
          ),
        ],
      ),
    );
  }
}

class CornDiseaseDetectionScreen extends StatefulWidget {
  @override
  _CornDiseaseDetectionScreenState createState() =>
      _CornDiseaseDetectionScreenState();
}

class _CornDiseaseDetectionScreenState
    extends State<CornDiseaseDetectionScreen> {
  CameraController? _cameraController;
  List<CameraDescription> cameras = [];
  bool _isModelReady = false;
  String _detectionResult = 'Waiting for detection...';
  double _accuracy = 0.0;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadModel();
  }

  void _initializeCamera() async {
    cameras = await availableCameras();
    _cameraController = CameraController(cameras[0], ResolutionPreset.medium);
    await _cameraController!.initialize();
    if (!mounted) {
      return;
    }
    setState(() {
      _isModelReady = true;
      _cameraController!.startImageStream((CameraImage image) {
        if (_isModelReady) {
          _detectDisease(image);
        }
      });
    });
  }

  void _loadModel() async {
    await Tflite.loadModel(
      model: 'assets/corn.tflite',
      labels: 'assets/cornlabels.txt',
    );
  }

  void _detectDisease(CameraImage image) async {
    if (!_isModelReady) {
      return;
    }

    List<dynamic>? output = await Tflite.runModelOnFrame(
      bytesList: image.planes.map((plane) {
        return plane.bytes;
      }).toList(),
      imageHeight: image.height,
      imageWidth: image.width,
      imageMean: 127.5,
      imageStd: 127.5,
      numResults: 2,
    );

    if (output != null && output.isNotEmpty) {
      setState(() {
        _detectionResult = output[0]['label'];
        _accuracy = output[0]['confidence'] * 100;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return DiseaseCard(
      diseaseName: 'Corn',
      detectionResult: _detectionResult,
      accuracy: _accuracy,
      cameraController: _cameraController,
    );
  }

  @override
  void dispose() {
    Tflite.close();
    _cameraController?.dispose();
    super.dispose();
  }
}

class PotatoDiseaseDetectionScreen extends StatefulWidget {
  @override
  _PotatoDiseaseDetectionScreenState createState() =>
      _PotatoDiseaseDetectionScreenState();
}

class _PotatoDiseaseDetectionScreenState
    extends State<PotatoDiseaseDetectionScreen> {
  CameraController? _cameraController;
  List<CameraDescription> cameras = [];
  bool _isModelReady = false;
  String _detectionResult = 'Waiting for detection...';
  double _accuracy = 0.0;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadModel();
  }

  void _initializeCamera() async {
    cameras = await availableCameras();
    _cameraController = CameraController(cameras[0], ResolutionPreset.medium);
    await _cameraController!.initialize();
    if (!mounted) {
      return;
    }
    setState(() {
      _isModelReady = true;
      _cameraController!.startImageStream((CameraImage image) {
        if (_isModelReady) {
          _detectDisease(image);
        }
      });
    });
  }

  void _loadModel() async {
    await Tflite.loadModel(
      model: 'assets/potato.tflite',
      labels: 'assets/potatolabels.txt',
    );
  }

  void _detectDisease(CameraImage image) async {
    if (!_isModelReady) {
      return;
    }

    List<dynamic>? output = await Tflite.runModelOnFrame(
      bytesList: image.planes.map((plane) {
        return plane.bytes;
      }).toList(),
      imageHeight: image.height,
      imageWidth: image.width,
      imageMean: 127.5,
      imageStd: 127.5,
      numResults: 2,
    );

    if (output != null && output.isNotEmpty) {
      setState(() {
        _detectionResult = output[0]['label'];
        _accuracy = output[0]['confidence'] * 100;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return DiseaseCard(
      diseaseName: 'Potato',
      detectionResult: _detectionResult,
      accuracy: _accuracy,
      cameraController: _cameraController,
    );
  }

  @override
  void dispose() {
    Tflite.close();
    _cameraController?.dispose();
    super.dispose();
  }
}

class RiceDiseaseDetectionScreen extends StatefulWidget {
  @override
  _RiceDiseaseDetectionScreenState createState() =>
      _RiceDiseaseDetectionScreenState();
}

class _RiceDiseaseDetectionScreenState
    extends State<RiceDiseaseDetectionScreen> {
  CameraController? _cameraController;
  List<CameraDescription> cameras = [];
  bool _isModelReady = false;
  String _detectionResult = 'Waiting for detection...';
  double _accuracy = 0.0;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadModel();
  }

  void _initializeCamera() async {
    cameras = await availableCameras();
    _cameraController = CameraController(cameras[0], ResolutionPreset.medium);
    await _cameraController!.initialize();
    if (!mounted) {
      return;
    }
    setState(() {
      _isModelReady = true;
      _cameraController!.startImageStream((CameraImage image) {
        if (_isModelReady) {
          _detectDisease(image);
        }
      });
    });
  }

  void _loadModel() async {
    await Tflite.loadModel(
      model: 'assets/rice.tflite',
      labels: 'assets/ricelabels.txt',
    );
  }

  void _detectDisease(CameraImage image) async {
    if (!_isModelReady) {
      return;
    }

    List<dynamic>? output = await Tflite.runModelOnFrame(
      bytesList: image.planes.map((plane) {
        return plane.bytes;
      }).toList(),
      imageHeight: image.height,
      imageWidth: image.width,
      imageMean: 127.5,
      imageStd: 127.5,
      numResults: 2,
    );

    if (output != null && output.isNotEmpty) {
      setState(() {
        _detectionResult = output[0]['label'];
        _accuracy = output[0]['confidence'] * 100;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return DiseaseCard(
      diseaseName: 'Rice',
      detectionResult: _detectionResult,
      accuracy: _accuracy,
      cameraController: _cameraController,
    );
  }

  @override
  void dispose() {
    Tflite.close();
    _cameraController?.dispose();
    super.dispose();
  }
}

class DiseaseCard extends StatelessWidget {
  final String diseaseName;
  final String detectionResult;
  final double accuracy;
  final CameraController? cameraController;

  DiseaseCard({
    required this.diseaseName,
    required this.detectionResult,
    required this.accuracy,
    required this.cameraController,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.all(16.0),
      child: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            if (cameraController != null &&
                cameraController!.value.isInitialized)
              AspectRatio(
                aspectRatio: 1,
                child: CameraPreview(cameraController!),
              ),
            SizedBox(height: 16.0),
            Text('Disease Type: $diseaseName'),
            Text('Detection Result: $detectionResult'),
            Text('Accuracy: ${accuracy.toStringAsFixed(2)}%'),
          ],
        ),
      ),
    );
  }
}
