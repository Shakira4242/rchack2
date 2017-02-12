module.exports = function(io) {
    var app = require('express');
    var router = app.Router();
    var fs = require('fs');
    var PythonShell = require('python-shell');

    router.get('/', function(req, res, next) {
      // while(true){
      //   // servo.value = servo.value - 1;
      //   // servo.to(servo.value);
      //   console.log(servo.value);
      // };
      res.render('index',{title: ""});
    });

    router.get('/home', function(req, res, next){
      res.render('home');
    });
    // router.post('/',function(req, res, next) {
    //   var response = req.body;
    //   setSteeringAngle(response.angle);
    // });


    var five = require("johnny-five")
      , board = new five.Board();
    
    var servo;
    var image;

    board.on("ready", function() {
      servo = new five.Servo({
        pin: 9
      });
      esc = new five.ESC({
        pin: 10
      });
      this.repl.inject({
        e: esc
      });
      // console.log("hello");
    });


    // set an angle between 0 - 180 degrees
    var setSteeringAngle = function(angle){
      servo.to(angle);
    }

    var predictSteeringAngle = function(image){
      return;
    }

    io.on('connection', function(socket) { 
        console.log("a user connected");

        // steering data sent from the browser
        socket.on('steeringAngle', function (data) {
          // change steering angle
          setSteeringAngle(data);
        }); 


        // socket.on('log', function(data){
        //   fs.appendFile('driving_dataset/data.txt', data["img"] + ".jpg" + " " + data["str"] + "\n", function(err){
        //     if (err) {

        //     }else { 
        //       console.log('File saved.');
        //     }
        //   });
        // });

        socket.on('img', function(data){
          fs.writeFile("driving_dataset/" + data["num"] + ".jpg", data["img"], 'base64', function(err){
            // console.log(err);
          });
        });

        socket.on('predict', function(data){
          PythonShell.run('predict.py', function (err, results) {
            if (err) throw err;
            console.log(results);
          });
        });
    });

    return router;
}