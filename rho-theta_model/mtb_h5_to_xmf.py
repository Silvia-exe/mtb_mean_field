import argparse
import h5py
import numpy as np
import os

def create_xdmf_for_paraview(h5_filename, xdmf_filename=None):
    if xdmf_filename is None:
        xdmf_filename = h5_filename.replace('.h5', '.xdmf')
    
    with h5py.File(h5_filename, 'r') as f:
        num_frames = f['o2'].shape[0]
        Lx = f.attrs['Lx']
        Ly = f.attrs['Ly']
        Nx = f.attrs['Nx']
        Ny = f.attrs['Ny']
    
    dx = Lx / (Nx - 1) if Nx > 1 else Lx
    dy = Ly / (Ny - 1) if Ny > 1 else Ly
    
    # Assume uniform time steps, dt=1.0
    times = ' '.join([str(i) for i in range(num_frames)])
    
    xdmf_content = f'''<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
      <Time TimeType="List">
        <DataItem Dimensions="{num_frames}" NumberType="Float" Precision="4" Format="XML">
          {times}
        </DataItem>
      </Time>
'''
    
    for t in range(num_frames):
        xdmf_content += f'''
      <Grid Name="Grid_t{t}" GridType="Uniform">
        <Topology TopologyType="2DRectMesh" NumberOfElements="{Ny} {Nx}"/>
        <Geometry GeometryType="ORIGIN_DXDY">
          <DataItem Dimensions="2" NumberType="Float" Precision="8" Format="XML">
            0.0 0.0
          </DataItem>
          <DataItem Dimensions="2" NumberType="Float" Precision="8" Format="XML">
            {dx} {dy}
          </DataItem>
        </Geometry>
        <Attribute Name="o2" AttributeType="Scalar" Center="Node">
          <DataItem ItemType="HyperSlab" Dimensions="{Ny} {Nx}" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" Format="XML">
              {t} 0 0
            </DataItem>
            <DataItem Dimensions="3" Format="XML">
              1 1 1
            </DataItem>
            <DataItem Dimensions="3" Format="XML">
              1 {Ny} {Nx}
            </DataItem>
            <DataItem Format="HDF">
              {os.path.basename(h5_filename)}:/o2
            </DataItem>
          </DataItem>
        </Attribute>
        <Attribute Name="bacteria" AttributeType="Scalar" Center="Node">
          <DataItem ItemType="HyperSlab" Dimensions="{Ny} {Nx}" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" Format="XML">
              {t} 0 0
            </DataItem>
            <DataItem Dimensions="3" Format="XML">
              1 1 1
            </DataItem>
            <DataItem Dimensions="3" Format="XML">
              1 {Ny} {Nx}
            </DataItem>
            <DataItem Format="HDF">
              {os.path.basename(h5_filename)}:/bacteria
            </DataItem>
          </DataItem>
        </Attribute>
        <Attribute Name="theta" AttributeType="Scalar" Center="Node">
          <DataItem ItemType="HyperSlab" Dimensions="{Ny} {Nx}" NumberType="Float" Precision="8">
            <DataItem Dimensions="3" Format="XML">
              {t} 0 0
            </DataItem>
            <DataItem Dimensions="3" Format="XML">
              1 1 1
            </DataItem>
            <DataItem Dimensions="3" Format="XML">
              1 {Ny} {Nx}
            </DataItem>
            <DataItem Format="HDF">
              {os.path.basename(h5_filename)}:/theta
            </DataItem>
          </DataItem>
        </Attribute>
      </Grid>
'''
    
    xdmf_content += '''
    </Grid>
  </Domain>
</Xdmf>
'''
    
    with open(xdmf_filename, 'w') as f:
        f.write(xdmf_content)
    
    print(f"XDMF file created: {xdmf_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create XDMF file for ParaView from HDF5.')
    parser.add_argument('filename', help='HDF5 file to process')
    parser.add_argument('--xdmf_filename', help='Output XDMF filename', default=None)
    args = parser.parse_args()

    create_xdmf_for_paraview(args.filename, xdmf_filename=args.xdmf_filename)
