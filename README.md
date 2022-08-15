# ebsd_denoising_autoencoder
![image](https://user-images.githubusercontent.com/79323764/184698864-61dd203a-7e19-499c-bd1e-b00d3b62351a.png)
This model takes input Kikuchi patterns of poor quality, and using weights trained on idealized Monte Carlo simulated patterns from EMSoft, and improves their image quality and band/peak contrast which leads to improved EBSD indexing and absolute strain determination. A copy of a paper demonstrating its use and its implications can be found via the following arxiv/DOI link: . If you're short on time or cannot access the paper, essentially this model takes Kikuchi patterns of low contrast and poor band definition, and improves them via an autoencoder trained on patterns with perfect contrast and band definition, thus adaptively improving the contrast and signal noise present in Kikuchi/EBSD diffraction patterns.

![image](https://user-images.githubusercontent.com/79323764/184698996-f8e7c3f0-5887-4439-b2f8-8f87194582c2.png)
A generalized workflow is given above, showing how the model was trained and how the model can be used to improve EBSPs. There are two portions of code which handle most of the heavy lifting: 'UP2_io.py' and 'denoise_from_model.py'. I work almost exclusively with EDAX equipment which uses .up1/.up2 as a way to store images in a single file, with the difference between them being bit depth: up1 files have 8-bit pixel values while up2 files have 16-bit ones. For newcomers to the EBSD world, this is preferable to having a directory with hundreds of thousands (typically less, but still...) of .jpg files and is typically less data/compute intensive. However, this means the images must be parsed from this single-file bitstream and Dave Rowenhorst, @drowenhorst-nrl, has done most of the groundwork writing up this code in IDL: I merely ported the UP2 file functionality to Python. UP2_io.py will do two things depending on what function you call. 'export_up2_jpg' will export every image in the UP2 file as a single .jpg file into a directory of your choosing, 'sep_image_path'. 'loose_2_up2' does the opposite, and takes a directory of loose jpgs and compiles them into a single UP2 file with a header which can be read by TSL OIM software. This is needed because the model only works from pattern-to-pattern, the output being a directory of denoised images, and reindexing in TSL OIM performs best (or often, at all) with UP2 files.

The code within 'denoise_from_model.py' will load in the pre-trained network and the associated weights, look through your directory of images, and denoise them using the forward prediction capability with Keras. There is functionality included from CV2 which can also use contrast limited adaptive histogram equalization, which can be applied before and/or after ML denoising to enhance the process, but the user must determine if this is appropriate for their dataset and what parameters to use. I've included an outline of the architecture below, note the input shape: 236x236. This corresponds with 2x2 binning on the EBSD detector, and is consequently what we trained the model on. If you are using different binning settings and thus pattern resolutions, image resizing would be required for compatibility. This resolution was, for us, a good compromise balancing dataset size, pattern fidelity/quality, indexability, and contains sufficient resolution for HR-EBSD strain analysis. The code used to train the network and load datasets is contained within 'train_model.py', and could be used to retrain the model to better suit your experimental needs. Although this model is alloy/microscope agnostic, as it is only looking at the diffraction pattern itself rather than Euler angles or phase, we recognize not all SEMs and detectors are equivalent, and you might need to tinker to get idealized results with the resolutions/patterns produced by your detector.

![image](https://user-images.githubusercontent.com/79323764/184703401-173f2836-1027-4ab2-a271-cf224d49942e.png)

Lastly, I'd like to thank Dave Rowenhorst at NRL (@drowenhorst-nrl), the David Fullwood Research Group @ BYU for their continued development of OpenXY (@BYU-MicrostructureOfMaterials), and Will Lenthe, Shawn Wallace, and Stuart Wright at EDAX for their work on OIM in general and their insight at Microscopy and Microanalysis 2022.
