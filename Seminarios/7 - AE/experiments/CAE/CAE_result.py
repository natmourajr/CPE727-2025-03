import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

def list_only_directories(path='.'):
    """Lists only the directories within a specified path."""
    directories = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            directories.append(entry)
    return sorted( directories )

def list_only_files(path='.'):
    """Lists only the files within a specified path."""
    files = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isfile(full_path):
            files.append(entry)
    return sorted( files )

def validate_image_set( imgs ):
    N = []
    for img in imgs:
        NE = img.split( '-' )
        N.append( int( NE[ 0 ]  ) )
    
    if len( set( N ) ) == 1:
        return True
    else:
        return False

def get_image_nbr( imgs ):
    N = []
    for img in imgs:
        NE = img.split( '-' )
        N.append( int( NE[ 0 ]  ) )
    
    if len( set( N ) ) == 1:
        return int( N[ 0 ] )
    else:
        return 0

def find_full_path( path, file ):
    for dpath, dnames, fnames in os.walk( path ):
        if file in fnames:
            return os.path.join( dpath, file )
    return ''

def get_frob_norm_error( file ):
    NE = file.split( '-' )
    return float( NE[ 1 ] )
    
def save_result( img_files ):
    N = len( img_files )
    imPos = []
    even = 0
    odd = 0

    # N = 16 -> imPos = [ 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15 ]
    for i in range( 0, N ):
        if ( i % 2 ):
            imPos.append( int( N / 2 ) + odd )
            odd = odd + 1
        else:
            imPos.append( even ) 
            even = even + 1

    fig = plt.figure(figsize=( N / 2, N / 8 ) )
    plt.axis( 'off' )
    plt.title( 'Contractive AE' )
    gs = gridspec.GridSpec( int( N / 8 ), int( N / 2 ) )
    gs.update(wspace=0.02, hspace=0.02)
    
    for i, file in enumerate( img_files ):
        # print( i, file )
        full_path = find_full_path( 'out', file )
        img = mpimg.imread( full_path )
        
        ax = plt.subplot( gs[ imPos[ i ] ] )
        ax.set_xticks( [] )
        ax.set_yticks( [] )
        ax.set_xticklabels( [] )
        ax.set_yticklabels( [] )

        if ( i % 2 ):
            ax.set_xlabel( str( get_frob_norm_error( file ) ) )
        if ( i == 0 ):
            ax.set_ylabel( 'input' )
        if ( i == 1 ):
            ax.set_ylabel( 'output' )
        ax.set_aspect( 'equal' )

        plt.imshow(img, cmap='Greys_r')
    # plt.show()
    idx = get_image_nbr( img_files )
    plt.savefig('plots/eval-{}.png'.format( str(idx).zfill(4) ), bbox_inches='tight')
    print( 'saving plots/eval-{}.png'.format( str(idx).zfill(4) ) )
    plt.close( fig )

#--------------------------------------------------------------------------------
X = []
Y = []
dirs = list_only_directories( 'out/' )
for dir in dirs:
    XY = dir.split( '-' )
    X.append( int( XY[ 0 ] ) )
    Y.append( float( XY[ 1 ] ) )

fig = plt.figure(figsize=(5, 4))
plt.axis( [ 0, 54, 50, 100 ] )
plt.plot( X, Y )
plt.xlabel( 'epochs' )
plt.ylabel( 'CAE loss' )
plt.title( 'training' ) 
plt.grid( True )

if not os.path.exists('plots/'):
    os.makedirs('plots/')
plt.savefig('plots/loss-epochs.png', bbox_inches='tight')
plt.close( fig )

dirs = list_only_directories( 'out/' )
files = []
for dir in dirs:
    files = files + list_only_files( 'out/{}'.format( dir ) )
files = sorted( files )

split_size = 2 * len( dirs )
f_split = [ files[ i : i + split_size ] for i in range( 0, len( files ), split_size ) ]

for res in f_split:
    res = res[ ::-1 ]
    if validate_image_set( res ) == False:
        continue

    save_result( res )
