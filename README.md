# Alizadeh

### کد U-Net 
در این فایل کد  یونت قرار داده شده است (unet_segmentation). در این کد برای راحتی استفاده از دیتاست های مختلف می توان آن دیتاست هایی که برای آموزش مدل نیاز است  را انتخاب کرده و باقی را کامنت کرد تا در مدل استفاده کرد. 
به عنوان نمونه: 
دو خط پایینی کامنت شده و از سه دیتاست و ماسک آنها که در دو خط بالا وجود دارند استفاده می شود.  

البته ممکن است نام این فایل ها بر اساس آپدیت کد تغییر کند.




img_folders = ['abyek_images', 'imgs_deliniation', 'original_bound_detection']

mask_folders = ['abyek_masks', 'masks_deliniation', 'img_bound_detection']

! # img_folders = ['abyek_images', 'original_bound_detection']

! # mask_folders = ['abyek_masks', 'img_bound_detection']

*** باقی مدل ها نیز به این قسمت اضافه شدند
