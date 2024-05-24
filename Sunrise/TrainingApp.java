package application;


import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import javax.inject.Inject;
import javax.inject.Named;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import com.kuka.common.ThreadUtil;
import com.kuka.generated.ioAccess.MediaFlangeIOGroup;
import com.kuka.roboticsAPI.applicationModel.RoboticsAPIApplication;
import static com.kuka.roboticsAPI.motionModel.BasicMotions.*;

import com.kuka.roboticsAPI.geometricModel.math.CoordinateAxis;
import com.kuka.roboticsAPI.geometricModel.math.Matrix;
import com.kuka.roboticsAPI.geometricModel.math.Transformation;
import com.kuka.roboticsAPI.geometricModel.math.Vector;

import com.kuka.roboticsAPI.conditionModel.ForceCondition;
import com.kuka.roboticsAPI.deviceModel.JointPosition;
import com.kuka.roboticsAPI.deviceModel.LBR;
import com.kuka.roboticsAPI.deviceModel.PositionInformation;
import com.kuka.roboticsAPI.geometricModel.AbstractFrame;
import com.kuka.roboticsAPI.geometricModel.CartDOF;
import com.kuka.roboticsAPI.geometricModel.Frame;
import com.kuka.roboticsAPI.geometricModel.ObjectFrame;
import com.kuka.roboticsAPI.geometricModel.Tool;
import com.kuka.roboticsAPI.geometricModel.Workpiece;
import com.kuka.roboticsAPI.geometricModel.World;
import com.kuka.roboticsAPI.geometricModel.math.Rotation;
import com.kuka.roboticsAPI.motionModel.controlModeModel.CartesianImpedanceControlMode;
import com.kuka.roboticsAPI.motionModel.controlModeModel.CartesianSineImpedanceControlMode;
import com.kuka.roboticsAPI.motionModel.controlModeModel.JointImpedanceControlMode;
import com.kuka.roboticsAPI.requestModel.SetManualOverrideRequest;
import com.kuka.roboticsAPI.uiModel.ApplicationDialogType;
import com.kuka.task.ITaskLogger;
import com.sun.org.apache.xpath.internal.operations.Bool;

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
public class TrainingApp extends RoboticsAPIApplication {

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

	// CHANGE FILE NAME IF NEEDED!
	String fileName = "parameters.txt";
	BufferedWriter _writer;
	boolean _finished = false;

	@Override
	public void initialize() {

		// Attach tool to flange
		_adapter.attachTo( _lbr.getFlange() );

		// Values to be optimized
		k_x = 600;
		k_y = 600;
		k_z = 600;
		k_A = 15;
		k_B = 150;
		k_C = 150;
		damp_trans = 0.7;
		damp_rot = 0.7;
		_stiff_params = new double[ 6 ];
		_damp_params = new double[ 2 ];

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
			_writer.write( "partialSuccess" );
			_writer.write( "\t" );
			_writer.newLine();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}


	@Override
	public void run() {

		// Turn on LED of Media Flange
		_myMediaFlange.setLEDBlue( true );

		// Begin with start configuration
		JointPosition startConfig = new JointPosition( Math.toRadians( -42.91 ), Math.toRadians( 84.58 ), Math.toRadians( 1 ), Math.toRadians( -71.79 ), Math.toRadians( 80.35 ), Math.toRadians( 35.83 ), Math.toRadians( -11.6 ) );


		boolean once = false;
		if( !once ){

			int count = 1;
			/*********** BEGIN LOOP *********************/

			/*********** UPDATE IMPEDANCE *********************/
			for( int groupKx = 0; groupKx < 3; groupKx++ ){
				for( int groupKy = 0; groupKy < 3; groupKy++ ){
					for( int groupKz = 0; groupKz < 3; groupKz++ ){
						for( int groupKA = 0; groupKA < 3; groupKA++ ){
							for( int groupKB = 0; groupKB < 3; groupKB++ ){
								for( int groupKC = 0; groupKC < 3; groupKC++ ){
									for( int groupDT = 0; groupDT < 3; groupDT++ ){
										for( int groupDR = 0; groupDR < 3; groupDR++ ){

											// Update impedance values
											_stiff_params[ 0 ] = (double) generateRandomKTrans( groupKx );
											_stiff_params[ 1 ] = (double) generateRandomKTrans( groupKy );
											_stiff_params[ 2 ] = (double) generateRandomKTrans( groupKz );
											_stiff_params[ 3 ] = (double) generateRandomKRot( groupKA );
											_stiff_params[ 4 ] = (double) generateRandomKRot( groupKB );
											_stiff_params[ 5 ] = (double) generateRandomKRot( groupKC );
											_damp_params[ 0 ] = generateRandomDampRa( groupDT );
											_damp_params[ 1 ] = generateRandomDampRa( groupDR );

											// Update controller
											updateController( );

											//											_logger.info( Double.toString( k_x ) );

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

											Transformation H_0_des = getApplicationData().getFrame( "/p_ori_fin_des" ).getTransformationFromParent();
											Rotation R_0_des = H_0_des.getRotation();
											Vector p_0_des = H_0_des.getTranslation(); 
											p_0_des = p_0_des.withZ( p_0_des.getZ() - 6.0 );

											Vector del_p_0_des = p_0_des.subtract( p_0_cur );
											Vector del_p_cur_des = R_cur_0_mat.multiply( del_p_0_des );

											Rotation R_cur_0 = Rotation.of( R_cur_0_mat );
											Rotation R_cur_des = R_cur_0.compose( R_0_des );		// R_cur_des = R_0_cur.transpose() * R_0_des;

											_adapter.move( linRel( Transformation.of( del_p_cur_des, R_cur_des ), _adapter.getFrame( "/TCP" ) ).setJointVelocityRel( _velRel_sl ).setMode( _cartImpCtrl ) );
											_adapter.move( linRel( 0, 0, -90, World.Current.getRootFrame() ).setJointVelocityRel( _velRel_slsl ).setMode( _sineImpMode ).breakWhen( _contactForceZ ) );
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

											boolean partialSuccess = false;
											if( _lbr.getCurrentCartesianPosition( _adapter.getFrame( "/TCP" ) ).getZ() < -180.0 )
											{
												partialSuccess = true;
											}

											// Lift gripper
											_adapter.move( linRel( 0.0, 0.0, 175.0, World.Current.getRootFrame() ).setJointVelocityRel( _velRel_sl ).setMode( _cartImpCtrl_z ) );


											/*********** WRITE TEXT FILE *********************/
											try {
												for( double param : _stiff_params ){
													_writer.write( Double.toString( param ) );
													_writer.write( "\t" );
												}
												for( double param : _damp_params ){
													_writer.write( Double.toString( param ) );
													_writer.write( "\t" );
												}
												_writer.write( Boolean.toString( success ) );
												_writer.write( "\t" );
												_writer.write( Boolean.toString( partialSuccess ) );
												_writer.newLine();
											} catch (IOException e) {
												// TODO Auto-generated catch block
												e.printStackTrace();
											}

											// Check for last iteration
											count = count + 1;

										}

									}

								}

							}

						}

					}

				}

			}

		} else{

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

			Transformation H_0_des = getApplicationData().getFrame( "/p_ori_fin_des" ).getTransformationFromParent();
			Rotation R_0_des = H_0_des.getRotation();
			Vector p_0_des = H_0_des.getTranslation(); 
			p_0_des = p_0_des.withZ( p_0_des.getZ() - 6.0 );

			Vector del_p_0_des = p_0_des.subtract( p_0_cur );
			Vector del_p_cur_des = R_cur_0_mat.multiply( del_p_0_des );

			Rotation R_cur_0 = Rotation.of( R_cur_0_mat );
			Rotation R_cur_des = R_cur_0.compose( R_0_des );		// R_cur_des = R_0_cur.transpose() * R_0_des;

			_adapter.move( linRel( Transformation.of( del_p_cur_des, R_cur_des ), _adapter.getFrame( "/TCP" ) ).setJointVelocityRel( _velRel_sl ).setMode( _cartImpCtrl ) );
			_adapter.move( linRel( 0, 0, -90, World.Current.getRootFrame() ).setJointVelocityRel( _velRel_slsl ).setMode( _sineImpMode ).breakWhen( _contactForceZ ) );
			_adapter.move( positionHold( _jointImpMode, 300, TimeUnit.MILLISECONDS ) );

			// Check for successful assembly
			if( _lbr.getCurrentCartesianPosition( _adapter.getFrame( "/TCP" ) ).getZ() < -200.0 )
			{
				_logger.info( "Success!" );
			}else{
				_logger.info( "Failure!" );
			}

			// Lift gripper
			_adapter.move( linRel( 0.0, 0.0, 175.0, World.Current.getRootFrame() ).setJointVelocityRel( _velRel_sl ).setMode( _cartImpCtrl_z ) );

		}


		//		}while( _finished = false );
		/*********** END LOOP *********************/


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


	public static int generateRandomKTrans( int group ){
		Random random = new Random();
		int kTransVal = 0;
		int low = 0;
		int high = 0;
		switch ( group ) {
		case 0: 
			// low value group
			low = 50;		// Lowest value of random function is inclusive
			high = 301;		// One int higher, since highest value of random function is exclusive
			kTransVal = random.nextInt( high - low ) + low;
			break;
		case 1:
			// med value group
			low = 301;					// Lowest value of random function is inclusive
			high = 701;					// One int higher, since highest value of random function is exclusive
			kTransVal = random.nextInt( high - low ) + low;
			break;
		case 2:
			// high value group;
			low = 701;						// Lowest value of random function is inclusive
			high = 1001;					// One int higher, since highest value of random function is exclusive
			kTransVal = random.nextInt( high - low ) + low;
			break;
		}
		return kTransVal;
	}


	public static int generateRandomKRot( int group ){
		Random random = new Random();
		int kRotVal = 0;
		int low = 0;
		int high = 0;
		switch ( group ) {
		case 0: 
			// low value group
			low = 5;		// Lowest value of random function is inclusive
			high = 11;		// One int higher, since highest value of random function is exclusive
			kRotVal = random.nextInt( high - low ) + low;
			break;
		case 1:
			// med value group
			low = 11;					// Lowest value of random function is inclusive
			high = 81;					// One int higher, since highest value of random function is exclusive
			kRotVal = random.nextInt( high - low ) + low;
			break;
		case 2:
			// high value group;
			low = 81;					// Lowest value of random function is inclusive
			high = 201;					// One int higher, since highest value of random function is exclusive
			kRotVal = random.nextInt( high - low ) + low;
			break;
		}
		return kRotVal;
	}


	public static double generateRandomDampRa( int group ){
		Random random = new Random();
		int dampRaInt = 0;
		double dampRa = 0.0;
		int low = 0;
		int high = 0;
		switch ( group ) {
		case 0: 
			// low value group
			low = 1;					// Lowest value of random function is inclusive
			high = 4;					// One int higher, since highest value of random function is exclusive
			dampRaInt = random.nextInt( high - low ) + low;
			dampRa = (double) dampRaInt/10;
			break;
		case 1:
			// med value group
			low = 4;					// Lowest value of random function is inclusive
			high = 8;					// One int higher, since highest value of random function is exclusive
			dampRaInt = random.nextInt( high - low ) + low;
			dampRa = (double) dampRaInt/10;
			break;
		case 2:
			// high value group;
			low = 8;					// Lowest value of random function is inclusive
			high = 10;					// One int higher, since highest value of random function is exclusive
			dampRaInt = random.nextInt( high - low ) + low;
			dampRa = (double) dampRaInt/10;
			break;
		}
		return dampRa;
	}


	public static void updateController( ){

		// Cart. Imp. controller
		_cartImpCtrl.parametrize( CartDOF.X ).setStiffness( _stiff_params[ 0 ] );
		_cartImpCtrl.parametrize( CartDOF.Y ).setStiffness( _stiff_params[ 1 ] );
		_cartImpCtrl.parametrize( CartDOF.Z ).setStiffness( _stiff_params[ 2 ] );
		_cartImpCtrl.parametrize( CartDOF.A ).setStiffness( _stiff_params[ 3 ] );
		_cartImpCtrl.parametrize( CartDOF.B ).setStiffness( _stiff_params[ 4 ] );
		_cartImpCtrl.parametrize( CartDOF.C ).setStiffness( _stiff_params[ 5 ] );
		_cartImpCtrl.parametrize( CartDOF.TRANSL ).setDamping( _damp_params[ 0 ] );
		_cartImpCtrl.parametrize( CartDOF.ROT ).setDamping( _damp_params[ 1 ] );
		_cartImpCtrl.setNullSpaceDamping( 0.7 );

		// Cart. Imp. controller with sine overlay
		_sineImpMode.parametrize( CartDOF.X ).setStiffness( _stiff_params[ 0 ] );
		_sineImpMode.parametrize( CartDOF.Y ).setStiffness( _stiff_params[ 1 ] );
		_sineImpMode.parametrize( CartDOF.Z ).setStiffness( _stiff_params[ 2 ] );
		_sineImpMode.parametrize( CartDOF.A ).setStiffness( _stiff_params[ 3 ] );
		_sineImpMode.parametrize( CartDOF.B ).setStiffness( _stiff_params[ 4 ] );
		_sineImpMode.parametrize( CartDOF.C ).setStiffness( _stiff_params[ 5 ] );
		_sineImpMode.parametrize( CartDOF.TRANSL ).setDamping( _damp_params[ 0 ] );
		_cartImpCtrl.parametrize( CartDOF.ROT ).setDamping( _damp_params[ 1 ] );
		_sineImpMode.parametrize( CartDOF.A ).setAmplitude( 0.6 );
		_sineImpMode.parametrize( CartDOF.A ).setFrequency( 4.5 );
		_sineImpMode.setNullSpaceDamping( 0.7 );

	}



}