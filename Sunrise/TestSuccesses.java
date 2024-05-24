package application;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

import javax.inject.Inject;
import javax.inject.Named;

import com.kuka.common.ThreadUtil;
import com.kuka.generated.ioAccess.MediaFlangeIOGroup;
import com.kuka.roboticsAPI.applicationModel.RoboticsAPIApplication;
import static com.kuka.roboticsAPI.motionModel.BasicMotions.*;

import com.kuka.roboticsAPI.conditionModel.ForceCondition;
import com.kuka.roboticsAPI.deviceModel.JointPosition;
import com.kuka.roboticsAPI.deviceModel.LBR;
import com.kuka.roboticsAPI.geometricModel.CartDOF;
import com.kuka.roboticsAPI.geometricModel.Tool;
import com.kuka.roboticsAPI.geometricModel.World;
import com.kuka.roboticsAPI.geometricModel.math.CoordinateAxis;
import com.kuka.roboticsAPI.geometricModel.math.Matrix;
import com.kuka.roboticsAPI.geometricModel.math.Rotation;
import com.kuka.roboticsAPI.geometricModel.math.Transformation;
import com.kuka.roboticsAPI.geometricModel.math.Vector;
import com.kuka.roboticsAPI.motionModel.controlModeModel.CartesianImpedanceControlMode;
import com.kuka.roboticsAPI.motionModel.controlModeModel.CartesianSineImpedanceControlMode;
import com.kuka.roboticsAPI.motionModel.controlModeModel.JointImpedanceControlMode;
import com.kuka.task.ITaskLogger;

/**
 * Implementation of a robot application.
 * <p>
 * The application provides a {@link RoboticsAPITask#initialize()} and a 
 * {@link RoboticsAPITask#run()} method, which will be called successively in 
 * the application lifecycle. The application will terminate automatically after 
 * the {@link RoboticsAPITask#run()} method has finished or after stopping the 
 * task. The {@link RoboticsAPITask#dispose()} method will be called, even if an 
 * exception is thrown during initialization or run. 
 * <p>
 * <b>It is imperative to call <code>super.dispose()</code> when overriding the 
 * {@link RoboticsAPITask#dispose()} method.</b> 
 * 
 * @see UseRoboticsAPIContext
 * @see #initialize()
 * @see #run()
 * @see #dispose()
 */
public class TestSuccesses extends RoboticsAPIApplication {

	@Inject
	ITaskLogger _logger;

	@Inject
	private LBR _lbr;

	@Inject
	private MediaFlangeIOGroup _myMediaFlange;

	@Inject
	@Named( "Adapter" )
	private Tool _adapter;

	// velocities
	double _velRel_slsl = 0.03;
	double _velRel_sl = 0.1;
	double _velRel_med = 0.35;
	double _velRel_fa = 0.45;

	// impedance controllers
	static CartesianImpedanceControlMode _cartImpCtrl;
	static CartesianImpedanceControlMode _cartImpCtrl_z;
	static CartesianSineImpedanceControlMode _sineImpMode;
	private JointImpedanceControlMode _jointImpMode;

	// force conditions
	private ForceCondition _contactForce;
	private ForceCondition _contactForceZ;

	// impedance parameters
	static double k_x;
	static double k_y;
	static double k_z;
	static double k_A;
	static double k_B;
	static double k_C;
	static double damp_trans;
	static double damp_rot;
	static double[] _stiff_params;
	static double[] _damp_params;

	// some stuff to store values
	FileReader fr;
	static Scanner _scanner;
	static int _numRows;
	static int _numColumns;
	static double _successes[][]; 
	
	String fileName = "parameters_circ.txt";
	BufferedWriter _writer;

	@Override
	public void initialize() {
		// initialize your application here

		// Attach tool to flange
		_adapter.attachTo( _lbr.getFlange() );

		// Values to be optimized
		k_x = 600;
		k_y = 600;
		k_z = 800;
		k_A = 5;
		k_B = 120;
		k_C = 120;
		damp_trans = 0.7;
		damp_rot = 0.7;

		// Cart. Imp. controller
		_cartImpCtrl = new CartesianImpedanceControlMode();
		_cartImpCtrl.parametrize( CartDOF.X ).setStiffness( k_x );
		_cartImpCtrl.parametrize( CartDOF.Y ).setStiffness( k_y );
		_cartImpCtrl.parametrize( CartDOF.Z ).setStiffness( k_z );
		_cartImpCtrl.parametrize( CartDOF.A ).setStiffness( k_A );
		_cartImpCtrl.parametrize( CartDOF.B ).setStiffness( k_B );
		_cartImpCtrl.parametrize( CartDOF.C ).setStiffness( k_C );
		_cartImpCtrl.parametrize( CartDOF.TRANSL ).setDamping( damp_trans );
		_cartImpCtrl.parametrize( CartDOF.ROT ).setDamping( damp_rot );
		_cartImpCtrl.setNullSpaceDamping( 0.7 );

		// Cart. Imp. controller with sine overlay
		_sineImpMode = new CartesianSineImpedanceControlMode();
		_sineImpMode.parametrize( CartDOF.X ).setStiffness( k_x );
		_sineImpMode.parametrize( CartDOF.Y ).setStiffness( k_y );
		_sineImpMode.parametrize( CartDOF.Z ).setStiffness( k_z );
		_sineImpMode.parametrize( CartDOF.A ).setStiffness( k_A );
		_sineImpMode.parametrize( CartDOF.B ).setStiffness( k_B );
		_sineImpMode.parametrize( CartDOF.C ).setStiffness( k_C );
		_sineImpMode.parametrize( CartDOF.TRANSL ).setDamping( damp_trans );
		_cartImpCtrl.parametrize( CartDOF.ROT ).setDamping( damp_rot );
		_sineImpMode.parametrize( CartDOF.A ).setAmplitude( 0.6 );
		_sineImpMode.parametrize( CartDOF.A ).setFrequency( 4.5 );
		_sineImpMode.setNullSpaceDamping( 0.7 );

		// Cart. Imp. controller for upwards motion
		_cartImpCtrl_z = new CartesianImpedanceControlMode();
		_cartImpCtrl_z.parametrize( CartDOF.X ).setStiffness( 350 );
		_cartImpCtrl_z.parametrize( CartDOF.Y ).setStiffness( 350 );
		_cartImpCtrl_z.parametrize( CartDOF.Z ).setStiffness( 500 );
		_cartImpCtrl_z.parametrize( CartDOF.A ).setStiffness( 20 );
		_cartImpCtrl_z.parametrize( CartDOF.B ).setStiffness( 35 );
		_cartImpCtrl_z.parametrize( CartDOF.C ).setStiffness( 35 );
		_cartImpCtrl_z.parametrize( CartDOF.TRANSL ).setDamping( 0.7 );
		_cartImpCtrl_z.parametrize( CartDOF.ROT ).setDamping( 0.7 );
		_cartImpCtrl_z.setNullSpaceDamping( 0.7 );

		// Joint impedance mode for position hold
		_jointImpMode = new JointImpedanceControlMode( 400, 400, 400, 400, 400, 400, 400 );

		// Force conditions
		_contactForce = ForceCondition.createSpatialForceCondition( _adapter.getDefaultMotionFrame(), 10 );
		_contactForceZ = ForceCondition.createNormalForceCondition( _adapter.getDefaultMotionFrame(), CoordinateAxis.Z, 55.0 );

		// File reading
		_numRows = 1431;
		_numColumns = 8;
		try {
			fr = new FileReader( "successfulTrial.txt" );
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		_scanner = new Scanner( fr );
		_successes = new double[_numRows][_numColumns];
		
		// File writing
		// Initialize buffer writer
		try {
			_writer = new BufferedWriter( new FileWriter( fileName ) );
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		// First line of text file
		try {
			_writer.write( "k_x" );
			_writer.write( "\t" );
			_writer.write( "k_y" );
			_writer.write( "\t" );
			_writer.write( "k_z" );
			_writer.write( "\t" );
			_writer.write( "k_A" );
			_writer.write( "\t" );
			_writer.write( "k_B" );
			_writer.write( "\t" );
			_writer.write( "k_C" );
			_writer.write( "\t" );
			_writer.write( "damp_t" );
			_writer.write( "\t" );
			_writer.write( "damp_r" );
			_writer.write( "\t" );
			_writer.write( "success" );
			_writer.write( "\t" );
			_writer.newLine();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	@Override
	public void run() {
		// your application execution starts here

		// Read file and store data in double-array
		readFile( );

		// Turn on LED of Media Flange
		_myMediaFlange.setLEDBlue( true );

		// Begin with start configuration
		JointPosition startConfig = new JointPosition( Math.toRadians( -42.39 ), Math.toRadians( 84.70 ), Math.toRadians( 1 ), Math.toRadians( -71.34 ), Math.toRadians( 80.34 ), Math.toRadians( 35.68 ), Math.toRadians( -10.99 ) );

		for( int i = 0; i < _numRows; i++ ){

			// Update controller
			updateController( _successes[i] );
			System.out.println( Arrays.toString( _successes[i] ) );

			/*********** DO MOTION *********************/
			_adapter.move( ptp( startConfig ).setJointVelocityRel( _velRel_fa ) );

			_adapter.move( linRel( 0.0, 0.0, -20.0, World.Current.getRootFrame() ).setJointVelocityRel( _velRel_slsl ) );

			// Touch the workpiece and stop when force is reached
			_adapter.move( linRel( -15.0, -20.0, -2.0, World.Current.getRootFrame() ).setJointVelocityRel( _velRel_slsl ).breakWhen( _contactForce ) );
			ThreadUtil.milliSleep( 200 );

			// Calculate the rotation to be aligned with the hole
			Transformation H_0_cur = _lbr.getCurrentCartesianPosition( _adapter.getFrame( "/TCP" ) ).getTransformationFromParent();
			Rotation R_0_cur = H_0_cur.getRotation();
			Matrix R_cur_0_mat = R_0_cur.getMatrix().transpose();
			Vector p_0_cur = H_0_cur.getTranslation();

			Transformation H_0_des = getApplicationData().getFrame( "/p_ori_fin_des_2" ).getTransformationFromParent();
			Rotation R_0_des = H_0_des.getRotation();
			Vector p_0_des = H_0_des.getTranslation(); 
			p_0_des = p_0_des.withZ( p_0_des.getZ() - 4.0 );

			Vector del_p_0_des = p_0_des.subtract( p_0_cur );
			Vector del_p_cur_des = R_cur_0_mat.multiply( del_p_0_des );

			Rotation R_cur_0 = Rotation.of( R_cur_0_mat );
			Rotation R_cur_des = R_cur_0.compose( R_0_des );		// R_cur_des = R_0_cur.transpose() * R_0_des;

			_adapter.move( linRel( Transformation.of( del_p_cur_des, R_cur_des ), _adapter.getFrame( "/TCP" ) ).setJointVelocityRel( _velRel_sl ).setMode( _cartImpCtrl ) );
			_adapter.move( linRel( 0, 0, -80, World.Current.getRootFrame() ).setJointVelocityRel( _velRel_slsl ).setMode( _sineImpMode ).breakWhen( _contactForceZ ) );
			_adapter.move( positionHold( _jointImpMode, 300, TimeUnit.MILLISECONDS ) );

			// Check for successful assembly
			boolean success = false;
			if( _lbr.getCurrentCartesianPosition( _adapter.getFrame( "/TCP" ) ).getZ() < -200.0 )
			{
				success = true;
				_logger.info( "Success!" );
			}else{
				_logger.info( "Failure!" );
			}

			// Lift gripper
			_adapter.move( linRel( 0.0, 0.0, 175.0, World.Current.getRootFrame() ).setJointVelocityRel( _velRel_sl ).setMode( _cartImpCtrl_z ) );
			
			
			/*********** WRITE TEXT FILE *********************/
			try {
				for( int k=0; k<_numColumns; k++ ){
					_writer.write( Double.toString( _successes[i][k] ) );
					_writer.write( "\t" );
				}
				_writer.write( Boolean.toString( success ) );
				_writer.newLine();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}
		
		// Close writer
		try {
			_writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// Disconnect LED of Media Flange
		_myMediaFlange.setLEDBlue( false );

	}		


	public static void updateController( double[] params ){

		// Cart. Imp. controller
		_cartImpCtrl.parametrize( CartDOF.X ).setStiffness( params[ 0 ] );
		_cartImpCtrl.parametrize( CartDOF.Y ).setStiffness( params[ 1 ] );
		_cartImpCtrl.parametrize( CartDOF.Z ).setStiffness( params[ 2 ] );
		_cartImpCtrl.parametrize( CartDOF.A ).setStiffness( params[ 3 ] );
		_cartImpCtrl.parametrize( CartDOF.B ).setStiffness( params[ 4 ] );
		_cartImpCtrl.parametrize( CartDOF.C ).setStiffness( params[ 5 ] );
		_cartImpCtrl.parametrize( CartDOF.TRANSL ).setDamping( params[ 6 ] );
		_cartImpCtrl.parametrize( CartDOF.ROT ).setDamping( params[ 7 ] );
		_cartImpCtrl.setNullSpaceDamping( 0.7 );

		// Cart. Imp. controller with sine overlay
		_sineImpMode.parametrize( CartDOF.X ).setStiffness( params[ 0 ] );
		_sineImpMode.parametrize( CartDOF.Y ).setStiffness( params[ 1 ] );
		_sineImpMode.parametrize( CartDOF.Z ).setStiffness( params[ 2 ] );
		_sineImpMode.parametrize( CartDOF.A ).setStiffness( params[ 3 ] );
		_sineImpMode.parametrize( CartDOF.B ).setStiffness( params[ 4 ] );
		_sineImpMode.parametrize( CartDOF.C ).setStiffness( params[ 5 ] );
		_sineImpMode.parametrize( CartDOF.TRANSL ).setDamping( params[ 6 ] );
		_cartImpCtrl.parametrize( CartDOF.ROT ).setDamping( params[ 7 ] );
		_sineImpMode.parametrize( CartDOF.A ).setAmplitude( 0.6 );
		_sineImpMode.parametrize( CartDOF.A ).setFrequency( 4.5 );
		_sineImpMode.setNullSpaceDamping( 0.7 );

	}

	public static void readFile( ){

		for( int i=0; i<_numRows; i++ ){
			for( int j=0; j<_numColumns; j++){
				
				if( _scanner.hasNextDouble() ){
					_successes[i][j] = _scanner.nextDouble();
				}
				else{
					System.err.println( "Not enough data in the file." );
				}
				
			}
		}
		
		_scanner.close();

	}

}