-sql-

CREATE TABLE specialities (
    id int NOT NULL, 
	name varchar2(102 char),
	code varchar2(8 char),   
	level_name varchar2(12 char) CHECK (level_name in ('Специалитет','Магистратура','Бакалавриат','Аспирантура')), --12
	level_id int CHECK (level_id in (1, 2, 3, 0)),	
	generation varchar2(10 char) CHECK (generation in ('ФГОС3+','ГОС2')), --6
	generation_id int CHECK (generation_id in (0,1)),	
	type varchar2(26 char) CHECK (type in ('направление','специальность')), --13
	type_id int CHECK (type_id in (0,1))
);

create sequence seq1 start with 500
increment by 1;

DROP SEQUENCE seq1;
DROP TABLE specialities;

-xsl-

<?xml version="1.0"?>
 <xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"> 
   <xsl:template match = "/"> 
    <html> 
    
    <body>
    
	<xsl:for-each select = "objects/object"> 
	<div> 
		INSERT INTO specialities (id, name, code, level_name, level_id, generation, generation_id, type, type_id) VALUES 
	</div> 
    <div>
	
    <xsl:text>(seq1.nextval,'</xsl:text>
    <xsl:value-of select="name"/>
	<xsl:text>','</xsl:text>
    <xsl:value-of select="code"/>
	<xsl:text>','</xsl:text>
    <xsl:value-of select="level"/>
	<xsl:text>','</xsl:text>
    <xsl:value-of select="level-id"/>
	<xsl:text>','</xsl:text>
    <xsl:value-of select="generation"/>
	<xsl:text>','</xsl:text>
    <xsl:value-of select="generation-id"/>
	<xsl:text>','</xsl:text>
    <xsl:value-of select="type"/>
	<xsl:text>','</xsl:text>
    <xsl:value-of select="type-id"/>
    <xsl:text>');</xsl:text>
	
    </div>
    </xsl:for-each>

   </body> 
   </html>   
   </xsl:template> 
</xsl:stylesheet>

-xml-

<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type='text/xsl' href='transitionscopy.xsl'?>
<objects type="array">
  <object>
    <id type="integer">307</id>
    <name>Технологии разделения изотопов и ядерное топливо</name>
    <code>14.05.03</code>
    <level>Специалитет</level>
    <level-id type="integer">1</level-id>
    <generation>ФГОС3+</generation>
    <generation-id type="integer">1</generation-id>
    <type>специальность</type>
    <type-id type="integer">1</type-id>
  </object>
  <object>
    <id type="integer">308</id>
    <name>Ядерные физика и технологии</name>
    <code>14.04.02</code>
    <level>Магистратура</level>
    <level-id type="integer">2</level-id>
    <generation>ФГОС3+</generation>
    <generation-id type="integer">1</generation-id>
    <type>направление</type>
    <type-id type="integer">0</type-id>
  </object>
</objects>

